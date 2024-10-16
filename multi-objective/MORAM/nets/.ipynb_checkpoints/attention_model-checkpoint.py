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
            node_dim = 3  # Number of features per block (size, query cost, request frequency)
            
            # Step context includes the block embedding and the remaining capacity
            step_context_dim = embedding_dim + 1  # +1 for remaining capacity
        
        else:
            assert problem.NAME == "block_selection", "Unsupported problem: {}".format(problem.NAME)
            
            # Same logic as above, step context should include the remaining capacity
            step_context_dim = embedding_dim + 1  # +1 for remaining capacity
            node_dim = 3  # Number of features per block (size, query cost, request frequency)
        
        # Embedding layers for the block features
        self.init_embed_qcobj = nn.Linear(1, embedding_dim)
        self.init_embed_mcobj = nn.Linear(2, embedding_dim)
        
        # Add a projection for the remaining capacity
        self.project_remaining_capacity = nn.Linear(1, embedding_dim)


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

    def forward(self, input, w_list, num_objs=2):
        """
        Forward pass of the AttentionModel. Processes the input data through the encoder and performs the decoding.
        
        :param input: (batch_size, ledger_size, 3) input node features
        :param w_list: list of weight vectors for objectives
        :param return_pi: if True, returns the sequence of selections (indices)
        :return: calculated costs, log likelihood, and the weighted objectives list
        """
        if self.checkpoint_encoder and self.training:
            # Checkpointing to save memory during training
            emb_list = []
            emb_list.append(checkpoint(self.embedder, self._init_embed(input[:, :, 0:1], "query_cost"))[0])
            emb_list.append(checkpoint(self.embedder, self._init_embed(input[:, :, 1:3], "monetary_cost"))[0])
            mixed_emb, _ = checkpoint(self.mix_gat, self.mix_emb(input))
            emb_list.append(mixed_emb)
        else:
            emb_list = []
            emb_list.append(self.embedder(self._init_embed(input[:, :, 0:1], "query_cost"))[0])
            emb_list.append(self.embedder(self._init_embed(input[:, :, 1:3], "monetary_cost"))[0])
            mixed_emb, _ = self.mix_gat(self.mix_emb(input))
            emb_list.append(mixed_emb)

        # Process weights for objectives
        w = torch.stack(w_list, dim=0).to(input.device)
        coef = self.w_net(w)
        
        # Prepare for mixed embeddings
        batch_size, ledger_size, hidden_dim = emb_list[0].shape
        coef_rep = coef.expand(batch_size, -1, -1)
        temp = torch.stack(emb_list, dim=-1)

        mixed = torch.einsum('bwo, bgho -> bwgh', coef_rep, temp).reshape(-1, ledger_size, hidden_dim)
        
        # Call the inner function to perform the decoding and get the selections
        _log_p, pi = self._inner(input, mixed, w)
        
        # Getting costs
        cost, mask, all_cost_list, total_storage = self.problem.get_costs(input, pi, w, num_objs, opts.max_capacity)

        # Calculate the log likelihood
        ll = self._calc_log_likelihood(_log_p, pi, mask)

        return cost, ll, all_cost_list, coef, pi, total_storage

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
        
        outputs = []
        selected_blocks = []
    
        # Expand the input based on the number of weight vectors
        input_rep = input.unsqueeze(1).expand(-1, w.size(0), -1, -1).reshape(-1, input.size(1), input.size(2))
        state = self.problem.make_state(input_rep, max_cap=opts.max_capacity)  # Initialize state with max capacity
    
        # Precompute the fixed context for attention
        fixed = self._precompute(mixed_embeddings)
        batch_size = state.ids.size(0)
    
        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished())
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Shrink the state if possible
                    state = state[unfinished]
                    fixed = fixed[unfinished]
    
            # Get the log probabilities and mask for valid selections
            log_p, mask = self._get_log_p(fixed, state, w)

            if mask[:, 0, :].all():
                break
                
            # Select the indices of the next blocks to store
            selected, masked_batches = self._select_block(log_p.exp()[:, 0, :], mask[:, 0, :])

            # Update the state with the selected blocks
            state = state.update(selected, masked_batches)
    
            # Adjust log_p and selected to match the original batch size if shrinking was applied
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)
                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_
    
            # Collect the log probabilities and selected blocks for this step
            outputs.append(log_p[:, 0, :])
            
            selected_blocks.append(selected)
    
            i += 1
            
        # Return the log probabilities and the selected blocks as a tensor
        return torch.stack(outputs, 1), torch.stack(selected_blocks, 1)



    def _calc_log_likelihood(self, _log_p, a, mask):
        """
        Calculate the log likelihood of selected actions.
    
        :param _log_p: Log probabilities of all possible actions (batch_size, num_steps, ledger_size)
        :param a: Tensor of selected actions (batch_size, num_steps)
        :param mask: Optional mask to ignore certain actions.
        :return: Summed log likelihood over selected actions.
        """
        # Create a mask for valid indices (where a != -1)
        valid_indices_mask = a != -1
    
        # Initialize log_p with zeros (same shape as a)
        log_p = torch.zeros(a.size(0), a.size(1), device=_log_p.device)
    
        # Only gather log probabilities for valid actions (ignore -1)
        if valid_indices_mask.any():
            # Use only valid indices for gathering
            valid_a = a.clone()
            valid_a[~valid_indices_mask] = 0  # Temporarily set invalid indices to 0 for gathering
            log_p[valid_indices_mask] = _log_p.gather(2, valid_a.unsqueeze(-1)).squeeze(-1)[valid_indices_mask]
    
        # Optional: mask out actions irrelevant to the objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0
    
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"
    
        # Calculate log_likelihood (sum over steps for each batch)
        return log_p.sum(1)



    
    def _get_log_p(self, fixed, state, w, normalize=True):
        
        temp = self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state, opts))
        # Compute query = context node embedding
        query = fixed.context_node_projected + temp

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any(), "NaN value detected in normalized log probs!"

        return log_p, mask

    
    def _get_parallel_step_context(self, embeddings, state, opts):
        """
        Returns the context per step for the block selection problem (BSP).
        
        :param embeddings: (batch_size, ledger_size, embed_dim) - the embeddings of the blocks.
        :param state: The current state of the block selection process.
        :param opts: Contains max_capacity and other parameters.
        :return: (batch_size, num_steps, context_dim) - concatenated embeddings and remaining capacity.
        """
        
        # Get the current node (block) that is being considered
        current_node = state.get_current_node().clone()  # (batch_size, num_steps)
        batch_size, num_steps = current_node.size()

        # Replace -1 (invalid selections) with the last valid selection
        invalid_mask = (current_node == -1)
        current_node[invalid_mask] = state.last_valid_a[invalid_mask]
    
        # Get embeddings for the current nodes
        current_embedding = torch.gather(
            embeddings,
            1,
            current_node.contiguous()
                .view(batch_size, num_steps, 1)
                .expand(batch_size, num_steps, embeddings.size(-1))
        )  # Shape: (batch_size, num_steps, embed_dim)
    
        # Compute remaining capacity
        remaining_capacity = (opts.max_capacity - state.stored_size[:, :, None])  # Shape: (batch_size, num_steps, 1)
    
        # Concatenate embeddings with remaining capacity
        return torch.cat((current_embedding, remaining_capacity), dim=-1)  # Shape: (batch_size, num_steps, context_dim)


    def _select_block(self, probs, mask):
        
        assert (probs == probs).all(), "Probs should not contain any NaNs"
        
        # Mask invalid actions by setting their probabilities to 0
        masked_probs = probs.clone()
        masked_probs[mask] = 0.0  # Set probabilities of masked actions to zero
    
        # Check if all actions are masked for any batch
        all_masked_batches = mask.all(dim=1)
        
        # Initialize the selected actions tensor
        selected = torch.full((masked_probs.size(0),), -1, dtype=torch.long).to(probs.device)
        
        # For batches that still have valid actions, perform multinomial or greedy sampling
        valid_batches = ~all_masked_batches
        if valid_batches.any():
            if self.decode_type == "greedy":
                masked_probs[mask] = -float('inf')  # Set probabilities of masked actions to zero
                _, selected[valid_batches] = masked_probs[valid_batches].max(1)
                # Assert no masked action was selected
                invalid_selections = mask[valid_batches].gather(1, selected[valid_batches].unsqueeze(-1)).squeeze(-1)
                if invalid_selections.any():
                    invalid_batches = invalid_selections.nonzero(as_tuple=True)[0]
                    print(f"Invalid selections in batches: {invalid_batches}")
                assert not invalid_selections.any(), "Decode greedy: infeasible action has maximum probability"
    
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

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, ledger_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            
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

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size) 
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))
    

        
        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, ledger_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
            
        if self.mask_logits:
            # Step 1: If the second 2000 dimension is not required, reduce or slice it
            mask = mask[:, 0, :]  # Taking the first slice of the second 2000 dimension, shape becomes [2000, 500]
            
            # Step 2: Add a singleton dimension to match the logits shape
            mask = mask.unsqueeze(1)  # Shape becomes [2000, 1, 500]

            logits[mask] = -1e9 # Use a large negative number instead of -inf

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
