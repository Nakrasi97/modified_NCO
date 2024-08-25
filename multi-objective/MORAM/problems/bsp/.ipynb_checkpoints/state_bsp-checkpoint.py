from typing import NamedTuple
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
    def initialize(loc, selected_dtype=torch.uint8):
        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return StateBlockSelection(
            loc=loc,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a=prev_a,
            selected_=torch.zeros(batch_size, 1, n_loc, dtype=selected_dtype, device=loc.device),
            stored_size=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
            selected_blocks=list()
        )
        
    def update(self, selected):
        prev_a = selected[:, None]
        self.selected_blocks.append(selected)
        stored_size = self.stored_size + self.loc.gather(1, selected[:, None, None].expand(-1, 1, self.loc.size(-1)))[:, :, 1]
    
        selected_ = self.selected_.scatter(-1, prev_a[:, :, None], 1)
    
        return self._replace(prev_a=prev_a, selected_=selected_, stored_size=stored_size, i=self.i + 1)
         
        
    def all_finished(self):
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        # Mask infeasible actions that would exceed the storage capacity
        return (self.selected > 0) | (self.stored_size + self.loc[..., 1] > self.max_cap)

    def construct_solutions(self, actions):
        return actions
