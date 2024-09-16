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
        print(n_loc)
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        print(f"Shape of prev_a at initialization: {prev_a.shape}")
        return StateBlockSelection(
            loc=loc,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a=prev_a,
            selected_=torch.zeros(batch_size, 1, n_loc, dtype=selected_dtype, device=loc.device),
            stored_size=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
            selected_blocks=list(),
            max_cap=max_cap
        )
        
    def update(self, selected, all_masked_batches):
        """
        Update the state after selecting blocks for storage.
        
        :param selected: indices of the selected blocks
        :return: updated state
        """
        # Ensure no invalid indices (-1) are passed to the gather operation
        valid_selected = selected.clone()
        if (valid_selected < 0).any():
            print(f"Invalid selections (e.g., -1) found: {valid_selected}")
            # Skip updates for these invalid selections
            valid_selected[valid_selected < 0] = 0  # Or handle as necessary
        
        prev_a = valid_selected[:, None, None]  # Shape: [batch size, 1, 1]
        print(f"Shape of prev_a: {prev_a.shape}")
        
        # Gather the relevant feature vector for the selected block
        gathered_loc = self.loc.gather(1, prev_a.expand(-1, -1, 4))  # Shape: [batch size, 1, 4]
        print(f"Shape of gathered_loc: {gathered_loc.shape}")
    
        # Update stored size with the size of the selected block (assuming it's feature index 1)
        print(f"Block sizes to add: {gathered_loc[:, :, 1].squeeze(1)}")
        print(f"Stored size before update: {self.stored_size}")
    
        # Only update stored size for batches that are NOT all masked and where a valid selection was made
        stored_size = self.stored_size.clone()
        valid_batches = (~all_masked_batches) & (selected != -1)
        if valid_batches.any():
            stored_size[valid_batches] += gathered_loc[valid_batches, :, 1].squeeze(1).unsqueeze(1)
        
        print(f"Stored size after update: {stored_size}")
    
        # Update the selected blocks
        selected_ = self.selected_.scatter(-1, prev_a, 1)
        print(f"Selected blocks: {selected_}")
    
        prev_a = prev_a.squeeze(-1)
    
        # Return the updated state
        return self._replace(prev_a=prev_a, selected_=selected_, stored_size=stored_size, i=self.i + 1)




         
        
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
