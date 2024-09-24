from typing import NamedTuple
from options import get_options
import torch

class StateBlockSelection(NamedTuple):
    loc: torch.Tensor
    ids: torch.Tensor
    prev_a: torch.Tensor
    selected_: torch.Tensor
    stored_size: torch.Tensor
    max_cap: float
    i: torch.Tensor
    selected_blocks: list
    last_valid_a: torch.Tensor

    # Retrieve options
    opts = get_options()
    
    @property
    def selected(self):
        return self.selected_

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                selected_=self.selected_[key],
                stored_size=self.stored_size[key],
            )
        return StateBlockSelection[key]

    @staticmethod
    def initialize(loc, max_cap=opts.max_capacity, selected_dtype=torch.uint8):
        """
        Initialize the state for the block selection problem.
        
        :param loc: tensor containing the block features
        :param max_cap: maximum storage capacity
        :return: initialized state
        """
        print("Initializing StateBlockSelection")
        batch_size, n_loc, _ = loc.size()
        
        return StateBlockSelection(
            loc=loc,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device),
            selected_=torch.zeros(batch_size, 1, n_loc, dtype=selected_dtype, device=loc.device),
            stored_size=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
            selected_blocks=list(),
            max_cap=max_cap,
            last_valid_a = torch.zeros(batch_size, 1, dtype=torch.long)
        )
        

    def update(self, selected, all_masked_batches):
        """
        Update the state after selecting blocks for storage.
    
        :param selected: indices of the selected blocks
        :return: updated state
        """
        # Print shapes for debugging
        print(f"Shape of loc: {self.loc.shape}")
        print(f"Shape of selected: {selected.shape}")
    
        # Prepare prev_a tensor
        prev_a = selected[:, None, None]  # Shape: [batch_size, 1, 1]
        print(f"Shape of prev_a: {prev_a.shape}")
        print(f"Prev_a: {prev_a}")
    
        # Identify valid selections where selected != -1
        valid_selections = (selected != -1)
        valid_indices = valid_selections.nonzero(as_tuple=True)[0]  # Indices of valid selections
    
        # Clone stored_size and selected_ to avoid in-place modifications
        stored_size = self.stored_size.clone()
        selected_ = self.selected_.clone()
        prev_a_squeezed = prev_a.squeeze(-1)  # Shape: [batch_size, 1]

        # Only update last_valid_a where valid selections (non -1) are made
        last_valid_a = self.last_valid_a.clone()
        last_valid_a[valid_selections] = selected[valid_selections].unsqueeze(1)
    
        if valid_indices.numel() > 0:
            # Gather data for valid selections
            prev_a_valid = prev_a[valid_selections]  # Shape: [num_valid, 1, 1]
            gathered_loc_valid = self.loc[valid_selections].gather(
                1, prev_a_valid.expand(-1, -1, self.loc.size(-1))
            )  # Shape: [num_valid, 1, loc_feature_size]
            print(f"Shape of gathered_loc_valid: {gathered_loc_valid.shape}")
    
            # Update stored_size for valid selections
            # Assuming size is at index 1 in the feature vector
            stored_size[valid_selections] += gathered_loc_valid[:, :, 1].squeeze(1).unsqueeze(1)
            print(f"Stored size after update: {stored_size}")
    
            # Update selected_ using advanced indexing
            prev_a_valid_squeezed = prev_a_valid.squeeze(-1).squeeze(-1)  # Shape: [num_valid]
            selected_[valid_indices, :, prev_a_valid_squeezed] = 1
            print(f"Selected blocks after update: {selected_}")
        else:
            print("No valid selections. Stored size remains unchanged.")
            print(f"Stored size: {stored_size}")
            print(f"Selected blocks: {selected_}")
    
        # Return the updated state
        return self._replace(prev_a=prev_a_squeezed, selected_=selected_, stored_size=stored_size, i=self.i + 1, last_valid_a=last_valid_a)


    
        
    def all_finished(self):
        """Check if all steps are finished."""
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        """Return the current node being considered."""
        return self.prev_a

    def get_mask(self):
        """
        Return a mask indicating valid blocks for selection.
        
        :return: mask tensor
        """
        print("Checking for selected blocks")
        # Check if the block has already been selected
        mask = (self.selected > 0)  # Selected blocks get True, others get False
        print(f"Shape of mask with only selected: {mask.shape}")

        print(f"Checking capacity violation")
        # Check if adding the block would exceed the storage capacity
        capacity_violation = (self.stored_size + self.loc[..., 1] > self.max_cap)
        # Ensure capacity_violation has the correct shape by unsqueezing it to match the mask's second dimension
        capacity_violation = capacity_violation.unsqueeze(1)  # Shape: [50, 1, 500]

        print("***Checking capacity violation")
        
        # Combine the two conditions
        mask = mask | capacity_violation
        
        return mask

    def construct_solutions(self, actions):
        return actions
