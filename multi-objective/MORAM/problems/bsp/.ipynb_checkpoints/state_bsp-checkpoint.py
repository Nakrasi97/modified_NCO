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
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
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
        
    def update(self, selected):
        """
        Update the state after selecting blocks for storage.
        
        :param selected: indices of the selected blocks
        :return: updated state
        """
        prev_a = selected[:, None, None]  # Shape: [2000, 1, 1]
        print(f"Shape of prev_a: {prev_a.shape}")
        
        # Gather the relevant feature vector for the selected block
        gathered_loc = self.loc.gather(1, prev_a)  # Shape: [2000, 1, 4]
        print(f"Shape of gathered_loc: {gathered_loc.shape}")
    
        # Update stored size with the size of the selected block (assuming it's feature index 1)
        stored_size = self.stored_size + gathered_loc[:, :, 1].squeeze(1)  # Shape: [2000]
        print(f"Stored blocks size shape: {stored_size.shape}")
    
        # Update the selected blocks
        selected_ = self.selected_.scatter(-1, prev_a, 1)
        print(f"Selected blocks shape: {selected_.shape}")
    
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
        # Check if the block has already been selected
        mask = (self.selected > 0)  # Selected blocks get True, others get False
        
        # Check if adding the block would exceed the storage capacity
        capacity_violation = (self.stored_size + self.loc[..., 1] > self.max_cap)
        
        # Combine the two conditions
        mask = mask | capacity_violation
        
        # Ensure the mask has the correct shape for broadcasting (e.g., [batch_size, 1, 1, ledger_size])
        # mask = mask.unsqueeze(0).unsqueeze(2)
        
        return mask

    def construct_solutions(self, actions):
        return actions
