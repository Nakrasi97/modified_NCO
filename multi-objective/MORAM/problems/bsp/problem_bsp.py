from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.bsp.state_bsp import stateBSP
from utils.beam_search import beam_search

import numpy as np
import matplotlib.pyplot as plt


class BSP(object):

    NAME = 'bsp'

    @staticmethod
    def get_costs(dataset, pi, w, scalar_func):
        
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        # d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))
        d = dataset.unsqueeze(1).expand(-1, w.size(0), -1, -1).reshape(-1, dataset.size(1), dataset.size(2))\
            .gather(1, pi.unsqueeze(-1).expand(-1, -1, dataset.size(-1)))
        
        cor1 = d[..., :2]
        cor2 = d[..., 2:]
        dist1 = (cor1[:, 1:] - cor1[:, :-1]).norm(p=2, dim=2).sum(1) + (cor1[:, 0] - cor1[:, -1]).norm(p=2, dim=1)
        dist2 = (cor2[:, 1:] - cor2[:, :-1]).norm(p=2, dim=2).sum(1) + (cor2[:, 0] - cor2[:, -1]).norm(p=2, dim=1)

        dist1 = dist1.reshape(-1, w.size(0))
        dist2 = dist2.reshape(-1, w.size(0))
        w_rep = w.unsqueeze(0).expand(dist1.size(0), -1, -1)

        dist = (w_rep * torch.stack([dist1, dist2], dim=-1))
        
        if scalar_func == "weighted-sum":
            return dist.sum(-1).detach(), None, [dist1, dist2]
        
        elif scalar_func == "tchebycheff": 
            return dist.max(-1)[0].detach(), None, [dist1, dist2]

    @staticmethod
    def make_dataset(*args, **kwargs):
        return BSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return stateBSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = BSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class BSPDataset(Dataset):
    
    def __init__(self, filename=None, size=500, num_samples=1000000, offset=0, distribution=None, correlation=0, block_features=4):
        super(BSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            self.data = torch.rand((num_samples, size, block_features))
            
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


    """ def load_rand_data(self, size, num_samples):
        path = './data/test200_instances_{}_mix3.pt'.format(size)
        if os.path.exists(path):
            pre_data = torch.load(path).permute(0, 2, 1)
            self.data = pre_data
            self.size = self.data.size(0)
            print(self.data.shape)
        else:
            print('Do not exist!', size, num_samples)
 """