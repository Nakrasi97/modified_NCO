import torch
import numpy as np
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches
from options import get_options

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many

from utils import load_problem
from time import sleep

opts = get_options()  # Get options to retrieve max_capacity

def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def log_mask_to_file(mask, step, file_path="mask_log.txt"):
    """
    Logs the mask at each decoding step to a file in a readable format.
    
    :param mask: Mask tensor to be logged
    :param step: Decoding step number
    :param file_path: Path to the log file
    """
    # Convert the mask tensor to a NumPy array
    mask_np = mask.cpu().numpy()  # Move the tensor to CPU if it's on GPU
    
    # Remove any singleton dimensions (like the middle 1) for better viewing
    mask_np = np.squeeze(mask_np)  # Removes dimensions of size 1
    
    # Open the file in append mode and write the mask with step information
    with open(file_path, "a") as f:
        f.write(f"Decoding step: {step}\n")
        f.write(f"Mask shape: {mask_np.shape}\n\n")
        
        # Write the mask in list format for each batch
        for batch_idx in range(mask_np.shape[0]):
            batch_mask = mask_np[batch_idx].tolist()  # Convert to a Python list
            f.write(f"Batch {batch_idx}: {batch_mask}\n")  # Write the mask as a list

        f.write("\n")  # Newline between steps for readability






class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return AttentionModelFixed[key]


class RoutingNet(nn.Module):
    def __init__(self, num_objs, hidden_dim):
        super(RoutingNet, self).__init__()

        self.embeding = nn.Linear(num_objs + 1, hidden_dim)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_dim, num_objs + 1)
        self.tanh = nn.Tanh()

    def forward(self, w):
        w = torch.cat([w, torch.zeros_like(w[:, 0].unsqueeze(-1), device=w.device)], dim=1)
        w_emb = self.embeding(w)
        w_emb = self.output(self.relu(w_emb)) + w
        w_emb = self.tanh(w_emb)
        return w_emb


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=True,
                 shrink_size=None,
                 num_objs=2
                 ):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        # Define the embedding layer for the block selection problem
        if problem.NAME == "block_selection":
            node_dim = 4  # Number of features per block (size, query cost, request frequency, transmission count)
            step_context_dim = embedding_dim
        else:
            assert problem.NAME == "block_selection", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = embedding_dim  # Embedding of first and last node
            node_dim = 4  # Number of features per block (size, query cost, request frequency, transmission count)

        self.init_embed_qcobj = nn.Linear(1, embedding_dim)
        self.init_embed_mcobj = nn.Linear(3, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.mix_emb = nn.Linear(node_dim, embedding_dim)

        self.mix_gat = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        self.w_net = RoutingNet(num_objs, hidden_dim)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, w_list, return_pi=False, num_objs=2):
        """
        Forward pass of the AttentionModel. Processes the input data through the encoder and performs the decoding.
        
        :param input: (batch_size, ledger_size, 4) input node features
        :param w_list: list of weight vectors for objectives
        :param return_pi: if True, returns the sequence of selections (indices)
        :return: calculated costs, log likelihood, and the weighted objectives list
        """
        if self.checkpoint_encoder and self.training:
            # Checkpointing to save memory during training
            emb_list = []
            emb_list.append(checkpoint(self.embedder, self._init_embed(input[:, :, 0:1], "query_cost"))[0])
            emb_list.append(checkpoint(self.embedder, self._init_embed(input[:, :, 1:4], "monetary_cost"))[0])
            mixed_emb, _ = checkpoint(self.mix_gat, self.mix_emb(input))
            emb_list.append(mixed_emb)
            print("Checkpointing...")
            print(f"Number of elements in embeddings list: {len(emb_list)}")
            print(f"Shape of tensors in embeddings list: {emb_list[0].shape}")
        else:
            emb_list = []
            emb_list.append(self.embedder(self._init_embed(input[:, :, 0:1], "query_cost"))[0])
            emb_list.append(self.embedder(self._init_embed(input[:, :, 1:4], "monetary_cost"))[0])
            mixed_emb, _ = self.mix_gat(self.mix_emb(input))
            emb_list.append(mixed_emb)
            print(f"Number of elements in embeddings list: {len(emb_list)}")
            print(f"Shape of tensors in embeddings list: {emb_list[0].shape}")

        # Process weights for objectives
        w = torch.stack(w_list, dim=0).to(input.device)
        print(f"Shape of weights tensor: {w.shape}")
        coef = self.w_net(w)
        print(f"Shape of weight embeddings tensor: {coef.shape}")
        
        # Prepare for mixed embeddings
        batch_size, ledger_size, hidden_dim = emb_list[0].shape
        coef_rep = coef.expand(batch_size, -1, -1)
        temp = torch.stack(emb_list, dim=-1)
        print(f"Before einsum, coef_rep.shape: {coef_rep.shape}, temp.shape: {temp.shape}")

        mixed = torch.einsum('bwo, bgho -> bwgh', coef_rep, temp).reshape(-1, ledger_size, hidden_dim)

        # Confirming the shape of the inputs to the decoder
        print(f"The shape of input: {input.shape}\nThe shape of mixed embeddings:{mixed.shape}\n The shape of weight vector: {w.shape}")
        
        # Call the inner function to perform the decoding and get the selections
        _log_p, pi = self._inner(input, mixed, w)
        
        print("Getting costs")
        cost, mask, all_cost_list = self.problem.get_costs(input, pi, w, num_objs, opts.max_capacity)

        # Calculate the log likelihood
        ll = self._calc_log_likelihood(_log_p, pi, mask)

        if return_pi:
            return cost, ll, pi

        return cost, ll, all_cost_list, coef

    def _init_embed(self, input, obj):
        """
        Initialize embeddings for the input nodes (blocks).
        
        :param input: (batch_size, ledger_size, specific_feature) input node features
        :return: embedded input features
        """
        if obj == "query_cost":
            return self.init_embed_qcobj(input)

        if obj == "monetary_cost":
            return self.init_embed_mcobj(input)


    def _precompute(self, mixed_embeddings, num_steps=1):
        """
        Precomputes the fixed context for the attention model.
        
        :param mixed_embeddings: (batch_size, ledger_size, embed_dim)
        :return: AttentionModelFixed - a named tuple containing precomputed tensors
        """
        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = mixed_embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(mixed_embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(mixed_embeddings, fixed_context, *fixed_attention_node_data)

    def _inner(self, input, mixed_embeddings, w):
        """
        Internal function to handle the decoding process for block selection.
        
        :param input: (batch_size, ledger_size, 4) input node features
        :param mixed_embeddings: mixed embeddings of the blocks
        :param w: weight vectors for objectives
        :return: log probabilities and indices of selected blocks
        """
        print("Starting _inner function")
        outputs = []
        selected_blocks = []
    
        # Expand the input based on the number of weight vectors
        print(f"Number of weight vectors: {w.size(0)}")
        input_rep = input.unsqueeze(1).expand(-1, w.size(0), -1, -1).reshape(-1, input.size(1), input.size(2))
        print(f"Input representation shape: {input_rep.shape}")
        state = self.problem.make_state(input_rep, max_cap=opts.max_capacity)  # Initialize state with max capacity
    
        # Precompute the fixed context for attention
        fixed = self._precompute(mixed_embeddings)
        batch_size = state.ids.size(0)
        print(f"Batch size: {batch_size}")
    
        # Perform decoding steps
        print("Start decoding")
        
        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            print("Decoding step", i)
            if self.shrink_size is not None:
                print("about to start!!")
                unfinished = torch.nonzero(state.get_finished())
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Shrink the state if possible
                    state = state[unfinished]
                    fixed = fixed[unfinished]
    
            # Get the log probabilities and mask for valid selections
            print("Get log probabilities and mask")
            log_p, mask = self._get_log_p(fixed, state, w)

            # Log the mask to file
            log_mask_to_file(mask, i, file_path="mask_log.txt")

            if mask[:, 0, :].all():
                print("All actions are masked. Ending decoding process.")
                print(f"Local blockchain storage: {state.stored_size}")
                break
                
            # Select the indices of the next blocks to store
            print("Select indices of next blocks to store")
            selected, masked_batches = self._select_block(log_p.exp()[:, 0, :], mask[:, 0, :])
            print(f"Shape of selected blocks: {selected.shape}")
            print(f"Selected blocks:{selected}")
            print("Selection complete")

            # Update the state with the selected blocks
            print("Update state with selected blocks")
            state = state.update(selected, masked_batches)
            print("State updated")

            # Check current local blockchain storage
            print(f"Local blockchain storage: {state.stored_size}")
    
            # Adjust log_p and selected to match the original batch size if shrinking was applied
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)
                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_
                print("Log probabilities adjusted to match original batch size")
    
            # Collect the log probabilities and selected blocks for this step
            outputs.append(log_p[:, 0, :])
            
            selected_blocks.append(selected)


            print("Selected blocks appended")
    
            i += 1
            
        print("Decoding completed")
        print("-------------------------------------------------------------------------------")
        print(f"Selected blocks: {selected_blocks}")
        print("-------------------------------------------------------------------------------")
        # Return the log probabilities and the selected blocks as a tensor
        return torch.stack(outputs, 1), torch.stack(selected_blocks, 1)



    def _calc_log_likelihood(self, _log_p, a, mask):

        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    
    def _get_log_p(self, fixed, state, w, normalize=True):
        print(f"Shape of node context embeddings before projection: {fixed.node_embeddings.shape}")
        temp = self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))
        print(f"Shape of temp: {temp.shape}")
        # Compute query = context node embedding
        print(f"Shape of context embedded: {fixed.context_node_projected.shape}")
        query = fixed.context_node_projected + temp

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
        print(f"Glimpse key shape:{glimpse_K.shape}\nGlimpse value shape: {glimpse_V.shape}\n Logit key shape:{logit_K.shape}")

        # Compute the mask
        mask = state.get_mask()
        print(f"What's the mask?: {mask.shape}")

        # Compute logits (unnormalized log_p)
        print(f"Query shape: {query.shape}")
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)
        print("Logits computation successful")

        # assert not torch.isnan(log_p).any(), "NaN value detected in logits!"
        # assert (log_p > 0).all(), "Non-positive value detected in logits!"

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)
            print("Log probabilities normalized")
            print(f"Shape of log_p: {log_p.shape}")

        assert not torch.isnan(log_p).any(), "NaN value detected in normalized log probs!"

        return log_p, mask

    
    def _get_parallel_step_context(self, embeddings, state):
        """
        Returns the context per step for the block selection problem (BSP).
        
        :param embeddings: (batch_size, ledger_size, embed_dim) - the embeddings of the blocks.
        :param state: The current state of the block selection process.
        :return: (batch_size, 1, context_dim)
        """
    
        # Get the current node (block) that is being considered
        current_node = state.get_current_node().clone()  # (batch_size, num_steps)
        print(f"Current node tensor shape: {current_node.shape}")
        batch_size, num_steps = current_node.size()

        # Replace -1 (invalid selections) with the last valid selection
        invalid_mask = (current_node == -1)
        current_node[invalid_mask] = state.last_valid_a[invalid_mask]
    
        # Gather the embedding of the current block 
        current_embedding = embeddings.gather(
            1,
            current_node.unsqueeze(-1).expand(batch_size, num_steps, embeddings.size(-1))
        )  # (batch_size, num_steps, embed_dim)
        print(f"Shape of current block embedding: {current_embedding.shape}")
        
        # For BSP, the context is simply the embedding of the current block
        return current_embedding

    
    def _select_block(self, probs, mask):
        
        assert (probs == probs).all(), "Probs should not contain any NaNs"
        
        # Mask invalid actions by setting their probabilities to 0
        masked_probs = probs.clone()
        masked_probs[mask] = 0.0  # Set probabilities of masked actions to zero
    
        # Check if all actions are masked for any batch
        all_masked_batches = mask.all(dim=1)
        
        # Initialize the selected actions tensor
        selected = torch.full((masked_probs.size(0),), -1, dtype=torch.long).to(probs.device)
    
        # Handle batches where all actions are masked by marking them as finished (-1)
        if all_masked_batches.any():
            print(f"Batches with all actions masked: {all_masked_batches.nonzero()}")
            # Batches where all actions are masked are already set to -1, so no need to do anything here
        
        # For batches that still have valid actions, perform multinomial or greedy sampling
        valid_batches = ~all_masked_batches
        if valid_batches.any():
            if self.decode_type == "greedy":
                _, selected[valid_batches] = masked_probs[valid_batches].max(1)
                assert not mask.gather(1, selected.unsqueeze(
                    -1)[valid_batches]).data.any(), "Decode greedy: infeasible action has maximum probability"
    
            elif self.decode_type == "sampling":
                selected[valid_batches] = masked_probs[valid_batches].multinomial(1).squeeze(1)
                # Check if sampling went OK, can go wrong due to bug on GPU
                max_resample_attempts = 10  # Arbitrary number of retries
                attempts = 0
                while mask.gather(1, selected.unsqueeze(-1)[valid_batches]).data.any():
                    print('Sampled bad values, resampling!')
                    selected[valid_batches] = masked_probs[valid_batches].multinomial(1).squeeze(1)
                    attempts += 1
                    if attempts >= max_resample_attempts:
                        print(f"Max resampling attempts reached for batch {valid_batches.nonzero()}")
                        break
    
        return selected, all_masked_batches


    
    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
        print(f"Key size: {key_size}")

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        print(f"Shape of glimpse: {glimpse_Q.shape}")

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, ledger_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        print(f"Compatibility shape before mask inner: {compatibility.shape}")
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            print(f"Shape of mask: {mask.shape}")
            
            # Step 1: Add batch dimension to the mask
            # Assuming mask is currently [2000, 2000, 500]
            exp_mask = mask.unsqueeze(0)  # Now the shape is [1, 2000, 2000, 500]
            
            # Step 2: If the second 2000 dimension is not necessary, slice or remove it
            # For example, if you only need the first 2000 dimension, you can slice it:
            exp_mask = exp_mask[:, :, 0, :]  # Slicing the second 2000 dimension, shape becomes [1, 2000, 500]
            
            # Step 3: Add singleton dimensions for broadcasting
            exp_mask = exp_mask.unsqueeze(2).unsqueeze(3)  # Now the shape is [1, 2000, 1, 1, 500]
            
            # Step 4: Expand the mask to match the batch size of 8
            exp_mask = exp_mask.expand(8, -1, -1, -1, -1)  # Now the shape is [8, 2000, 1, 1, 500]
            
            # Now the mask shape matches the compatibility shape and can be used in operations

            #compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf
            #expanded_mask = mask.expand_as(compatibility)
            compatibility[exp_mask] = -1e9 # Use a large negative number instead of -inf
            print(f"Compatibility shape after mask inner: {compatibility.shape}")

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size) 
        print("Computing attention heads")
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)
        print(f"Heads shape: {heads.shape}")

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))
        print(f"Glimpse shape: {glimpse.shape}")

        
        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, ledger_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))
        print(f"Logits shape: {logits.shape}")

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
            print("Logits clipping successful")
        if self.mask_logits:
            # Step 1: If the second 2000 dimension is not required, reduce or slice it
            mask = mask[:, 0, :]  # Taking the first slice of the second 2000 dimension, shape becomes [2000, 500]
            
            # Step 2: Add a singleton dimension to match the logits shape
            mask = mask.unsqueeze(1)  # Shape becomes [2000, 1, 500]

            logits[mask] = -1e9 # Use a large negative number instead of -inf
            print("Masking successful")

        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):
        # For the Block Selection Problem (BSP), we simply return the fixed data.
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, ledger_size, head_dim)
        )
