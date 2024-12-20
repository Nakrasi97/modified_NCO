import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils.functions import move_to
from plots import plot_pareto_subplots, plot_comparisons
from nsga import nsga2bsp, nsga3bsp

from pymoo.indicators.hv import HV
import numpy as np


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


# Validate
import time

def validate(model, val_dataset, opts):
    print('Validating...')

    # Initialize dictionary to store the results
    results = {
        'model': {'query_costs': [], 'monetary_costs': [], 'local_storage': [], 'runtime': 0},
        'nsga2': {'query_costs': [], 'monetary_costs': [], 'local_storage': [], 'runtime': 0},
        'nsga3': {'query_costs': [], 'monetary_costs': [], 'local_storage': [], 'runtime': 0}
    }

    # Measure the model's runtime
    start_time = time.time()

    # Call rollout to get the model's results
    cost, all_objs_list, blocks_selected, storage_used = rollout(model, val_dataset, opts)

    # Store the model runtime
    results['model']['runtime'] = time.time() - start_time

    print(f"Model runtime: {results['model']['runtime']} seconds")
    
    # Stack the objectives (query cost, monetary cost)
    all_objs = torch.stack(all_objs_list, dim=2)  # Shape: [batch_size, num_weights, 2]

    # Prepare reference point for HV calculation (worst-case objectives)
    worst_query_cost = all_objs[..., 0].min()  # Since it's maximization, min() is the worst
    worst_monetary_cost = all_objs[..., 1].min()

    ref_query = worst_query_cost - (0.5 * abs(worst_query_cost))
    ref_monetary = worst_monetary_cost - (0.5 * abs(worst_monetary_cost))
    
    reference_point = torch.tensor([ref_query, ref_monetary], device=opts.device)
    hv_fn = HV(ref_point=reference_point.cpu().numpy())

    # Initialize tensor to hold the nondominated set
    batch_size = all_objs.size(0)
    NDS = torch.full_like(all_objs, float('nan'))

    # Determine the nondominated solutions
    for i in range(batch_size):
        dominated = torch.zeros(opts.num_weights, dtype=torch.bool, device=opts.device)
        for j in range(opts.num_weights):
            for k in range(opts.num_weights):
                if j != k:
                    is_dominated = ((all_objs[i, k, :] >= all_objs[i, j, :]).all() and 
                                    (all_objs[i, k, :] > all_objs[i, j, :]).any())
                    if is_dominated:
                        dominated[j] = True
                        break
        nondominated_mask = ~dominated
        NDS[i, nondominated_mask, :] = all_objs[i, nondominated_mask, :]


    # Calculate hypervolume
    hv_list = []
    
    for i in range(batch_size):
        # Filter out NaNs (which correspond to dominated or invalid solutions)
        valid_solutions = NDS[i][~torch.isnan(NDS[i][:, 0])]
        
        if valid_solutions.size(0) > 0:
            # Negate the objectives for hypervolume calculation in a maximization problem
            valid_solutions_minimized = -valid_solutions.cpu().numpy()
            
            # Compute the hypervolume for the valid, nondominated set
            hv = hv_fn.do(valid_solutions_minimized)
            hv_list.append(hv)

            # Apply the nondominated mask to the storage for this batch
            nondominated_mask = ~torch.isnan(NDS[i][:, 0])

            # Calculate local storage used for nondominated solutions
            nondominated_storage = storage_used[i][nondominated_mask].cpu().numpy()  # Use the i-th sample's storage info
            
            # Store valid costs
            query_costs = valid_solutions[:, 0].cpu().numpy()
            monetary_costs = valid_solutions[:, 1].cpu().numpy()
            results['model']['query_costs'].extend(query_costs)
            results['model']['monetary_costs'].extend(monetary_costs)
            results['model']['local_storage'].extend(nondominated_storage)
    
    # Calculate final HV statistics (if there are valid hypervolumes)
    if len(hv_list) > 0:
        all_hv = torch.tensor(hv_list)
        print(f"HV values: {all_hv}")
        print('Mean HV for MORAM: {:.3f} +- {:.3f}'.format(all_hv.mean().item(), torch.std(all_hv).item()))
    else:
        print("No valid hypervolume calculated.")
        
            
    # Run NSGA-II and NSGA-III on the same dataset
    if opts.compare_nsga:
        nsga2_soln = []
        nsga3_soln = []

        # Measure NSGA-II runtime
        start_time = time.time()
        for i in range(batch_size):
            nsga2_costs, nsga2_storage = nsga2bsp(val_dataset.gen_blocks[i])
            nsga2_soln.append(nsga2_costs)
            results['nsga2']['query_costs'].extend(nsga2_costs[:, 0])
            results['nsga2']['monetary_costs'].extend(nsga2_costs[:, 1])
            results['nsga2']['local_storage'].extend(nsga2_storage)
        results['nsga2']['runtime'] = time.time() - start_time

        print(f"NSGA2 runtime: {results['nsga2']['runtime']} seconds")

        # Measure NSGA-III runtime
        start_time = time.time()
        for i in range(batch_size):
            nsga3_costs, nsga3_storage = nsga3bsp(val_dataset.gen_blocks[i])
            nsga3_soln.append(nsga3_costs)
            results['nsga3']['query_costs'].extend(nsga3_costs[:, 0])
            results['nsga3']['monetary_costs'].extend(nsga3_costs[:, 1])
            results['nsga3']['local_storage'].extend(nsga3_storage)
        results['nsga3']['runtime'] = time.time() - start_time

        print(f"NSGA3 runtime: {results['nsga3']['runtime']} seconds")

    # Plot comparisons between the model, NSGA-II, and NSGA-III
    plot_comparisons(results)

    return cost, all_objs_list, NDS, all_hv, None



def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")

    def eval_model_bat(bat, model):
        with torch.no_grad():
            cost, _, all_objs, _, selections, st = model(
                move_to(bat, opts.device),
                opts.w_list,
                num_objs=opts.num_objs
            )
        return cost.data.cpu(), all_objs, selections, st

    cost_list = []
    obj_list = []
    storage_list = []  # Track storage usage
    
    for o in range(opts.num_objs):
        obj_list.append([])
    for bat in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar):
        cost, all_objs, sel, loc_st = eval_model_bat(bat, model)
        cost_list.append(cost)
        storage_list.append(loc_st)
        for o in range(opts.num_objs):
            obj_list[o].append(all_objs[o])
            

    
    # Stack storage used across batches and return it
    return torch.cat(cost_list, 0), [torch.cat(obj_list[o], 0) for o in range(opts.num_objs)], None, torch.cat(storage_list, 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """

    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]

    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
            num_blocks=opts.ledger_size,
            num_samples=opts.epoch_size
        )
    )

    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0, pin_memory=True)
    
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    print("Starting training loop...")
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        print(f"Processing batch {batch_id}...")
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward, all_objs, NDS, HV, num_NDS = validate(model, val_dataset, opts)
    if not opts.no_tensorboard:
        for i in range(opts.num_weights):
            opts.logger_list[i].log_value('val_avg_reward', avg_reward[:, i].mean().item(), step)
            for j in range(opts.num_objs):
                opts.logger_list[i].log_value('val_dist{}'.format(j), all_objs[j][:, i].mean().item(), step)
        tb_logger.log_value('HV', HV.mean().item(), step)
        print('Epoch{} HV: '.format(epoch), HV.mean().item())

    if lr_scheduler.get_last_lr()[0] > opts.lr_critic:
        lr_scheduler.step()


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    set_decode_type(model, "sampling")
    x = move_to(batch, opts.device)
    cost, log_likelihood, all_costs, coef, sels, st = model(x, opts.w_list, num_objs=opts.num_objs)
    log_likelihood = log_likelihood.reshape(-1, opts.num_weights)

    obj_tensor = torch.stack(all_costs, dim=2).unsqueeze(1).expand(-1, opts.num_weights, -1, -1)
    if torch.cuda.device_count() > 1:
        w_tensor = torch.stack(opts.w_list, dim=0)[:, :opts.num_objs].unsqueeze(0).unsqueeze(2).expand_as(obj_tensor).to(obj_tensor.device)
    else:
        w_tensor = torch.stack(opts.w_list, dim=0).unsqueeze(0).unsqueeze(2).expand_as(obj_tensor).to(obj_tensor.device)
    
    # Sort in descending order since we want to maximize
    score = (w_tensor * obj_tensor).sum(-1).sort(descending=True)[0]
    
    # Adjust the REINFORCE loss for maximization
    reinforce_loss = ((score[:, :, :opts.num_top].mean(-1) - cost) * log_likelihood)
    
    # Loss and backpropagation
    loss = reinforce_loss
    optimizer.zero_grad()
    loss.mean().backward()

    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, loss, reinforce_loss, all_costs, coef, tb_logger, opts)


