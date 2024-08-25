import torch
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
            node_dim = 4  # Number of features per block (size, query cost, request frequency, transmission count)
            step_context_dim = embedding_dim
        else:
            assert problem.NAME == "block_selection", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = embedding_dim  # Embedding of first and last node
            node_dim = 4  # Number of features per block (size, query cost, request frequency, transmission count)

        self.init_embed = nn.Linear(node_dim, embedding_dim)

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

        self.w_net = RoutingNet(num_objs, hidden_dim)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, w_list, return_pi=False, num_objs=2):
        """
        :param input: (batch_size, ledger_size, 4) input node features
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """
    
        # Checkpointing for memory efficiency
        if self.checkpoint_encoder and self.training:
            emb_list = []
            # Initialize embeddings for the block features
            emb_list.append(checkpoint(self.embedder, self.init_embed(input[:, :, :]))[0])
        else:
            emb_list = []
            # Initialize embeddings for the block features
            emb_list.append(self.embedder(self.init_embed(input[:, :, :]))[0])
    
        # Weight tensor processing
        w = torch.stack(w_list, dim=0).to(input.device)
        coef = self.w_net(w)
    
        # Ensure embeddings align with coef
        batch_size, ledger_size, hidden_dim = emb_list[0].shape
        coef_rep = coef.expand(batch_size, -1, -1)
        temp = torch.stack(emb_list, dim=-1)
        mixed = torch.einsum('bwo, bgho -> bwgh', coef_rep, temp).reshape(-1, ledger_size, hidden_dim)
    
        # Internal processing
        _log_p, pi = self._inner(input, mixed, w)
    
        # Retrieve options
        opts = get_options()
    
        # Cost computation for block selection
        cost, mask, all_dist_list = self.problem.get_costs(input, pi, w, num_objs, opts.max_capacity)
    
        # Log likelihood calculation
        ll = self._calc_log_likelihood(_log_p, pi, mask)
    
        if return_pi:
            return cost, ll, pi
    
        return cost, ll, all_dist_list, coef


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
        outputs = []
        selected_blocks = []
    
        input_rep = input.unsqueeze(1).expand(-1, w.size(0), -1, -1).reshape(-1, input.size(1), input.size(2))
        state = self.problem.make_state(input_rep)
    
        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(mixed_embeddings)
    
        batch_size = state.ids.size(0)
    
        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and state.all_finished()):
    
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                # Check if we can shrink by at least shrink_size and if this leaves at least 16
                # (otherwise batch norm will not work well and it is inefficient anyway)
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    # Filter states
                    state = state[unfinished]
                    fixed = fixed[unfinished]
    
            log_p, mask = self._get_log_p(fixed, state, w)
    
            # Select the indices of the next blocks to store, result (batch_size) long
            selected = self._select_block(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
    
            state = state.update(selected)
    
            # Now make log_p, selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)
    
                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_
    
            # Collect output of step
            outputs.append(log_p[:, 0, :])
            selected_blocks.append(selected)
    
            i += 1
    
        # Combine all selected blocks into a final selection mask or list of indices
        final_selection = torch.stack(selected_blocks, dim=1)  # Shape: (batch_size, num_steps)
    
        # Convert this selection into a binary mask where selected blocks are marked with 1
        # If you want a list of selected block indices instead, you can return final_selection directly
        selection_mask = torch.zeros_like(log_p).scatter_(1, final_selection, 1)
    
        # Return log probabilities and the selection mask
        return torch.stack(outputs, 1), selection_mask


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
        temp = self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))
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

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state):
        """
        Returns the context per step for the block selection problem (BSP).
        
        :param embeddings: (batch_size, ledger_size, embed_dim) - the embeddings of the blocks.
        :param state: The current state of the block selection process.
        :return: (batch_size, 1, context_dim)
        """
    
        # Get the current node (block) that is being considered
        current_node = state.get_current_node()  # (batch_size, num_steps)
        batch_size, num_steps = current_node.size()
    
        # Gather the embedding of the current block
        current_embedding = embeddings.gather(
            1,
            current_node.unsqueeze(-1).expand(batch_size, num_steps, embeddings.size(-1))
        )  # (batch_size, num_steps, embed_dim)
    
        # For BSP, the context is simply the embedding of the current block
        return current_embedding

    def _select_block(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(
                -1)).data.any(), "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, ledger_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

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
            logits[mask] = -math.inf

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
