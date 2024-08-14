from typing import NamedTuple
import torch

class StateBlockSelection(NamedTuple):
    loc: torch.Tensor
    ids: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor
    lengths: torch.Tensor
    i: torch.Tensor
    visited_blocks: list

    @property
    def visited(self):
        return self.visited_

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
            )
        return StateBlockSelection[key]

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):
        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return StateBlockSelection(
            loc=loc,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            prev_a=prev_a,
            visited_=torch.zeros(batch_size, 1, n_loc, dtype=visited_dtype, device=loc.device),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
            visited_blocks=list()
        )

    def update(self, selected):
        prev_a = selected[:, None]
        self.visited_blocks.append(selected)
        lengths = self.lengths

        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)

        return self._replace(prev_a=prev_a, visited_=visited_,
                             lengths=lengths, i=self.i + 1)

    def all_finished(self):
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited > 0

    def construct_solutions(self, actions):
        return actions
