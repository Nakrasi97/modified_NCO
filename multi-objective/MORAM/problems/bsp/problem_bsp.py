from torch.utils.data import Dataset
import torch
from problems.bsp.state_bsp import StateBlockSelection
from utils.beam_search import beam_search
import simpy

import numpy as np
import matplotlib.pyplot as plt

class Block:
    def __init__(self, index, size):
        self.index = index
        self.size = size
        self.query_freq = 0
        self.query_cost = 0
        self.transmission_count = 0

    def update_query_cost(self, conn_bandwidth):
        self.query_cost += (self.size * 8) / conn_bandwidth
        self.query_freq += 1
        self.transmission_count += 1

class BlockDataset(Dataset):
    def __init__(self, num_samples=1000, num_blocks=50, min_size=1, max_size=10, mean_size=5, std_dev=2, duration=60):
        super(BlockDataset, self).__init__()

        self.num_samples = num_samples
        self.num_blocks = num_blocks
        self.data = []
        
        for _ in range(num_samples):
            blocks = self.generate_blocks(num_blocks, min_size, max_size, mean_size, std_dev)
            blocks = self.simulate_queries_with_simpy(blocks, duration)
            block_data = self.extract_block_data(blocks)
            self.data.append(block_data)
        
        self.size = len(self.data)

    def generate_blocks(self, num_blocks, min_size, max_size, mean_size, std_dev):
        blocks = []

        for index in range(num_blocks):
            size = np.clip(round(np.random.normal(mean_size, std_dev), 2), a_min=min_size, a_max=max_size)
            blocks.append(Block(index, size))

        return blocks

    def simulate_queries_with_simpy(self, blocks, duration):
        conn_bandwidth = 100  # in Mbps
        delay = 0.005  # seconds

        def query_process(env, block):
            while True:
                block.update_query_cost(conn_bandwidth)
                yield env.timeout(delay)

        env = simpy.Environment()
        for block in blocks:
            env.process(query_process(env, block))
        
        env.run(until=duration * 60)
        return blocks

    def extract_block_data(self, blocks):
        block_data = []
        for block in blocks:
            block_info = [block.query_cost, block.size, block.query_freq, block.transmission_count]
            block_data.append(block_info)
        return torch.FloatTensor(block_data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]




class BlockSelectionProblem(object):
    NAME = 'block_selection'

    @staticmethod
    def get_costs(dataset, pi, w, num_objs):
        # Ensure the tour or selection is valid
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid block selection"

        # Gather dataset in order of selected blocks
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Assume first column is query cost, second is block size, third is request frequency, fourth is transmission count
        query_cost = d[..., 0]
        block_size = d[..., 1]
        request_freq = d[..., 2]
        transmission_count = d[..., 3]

        # Calculate the total query cost
        total_query_cost = query_cost.sum(1)
        
        # Calculate the total monetary cost
        storage_cost = w[:, 0].unsqueeze(1) * block_size.sum(1)
        request_cost = w[:, 1].unsqueeze(1) * request_freq.sum(1)
        transmission_cost = w[:, 2].unsqueeze(1) * (block_size * transmission_count).sum(1)
        total_monetary_cost = storage_cost + request_cost + transmission_cost

        if num_objs == 2:
            return torch.stack([total_query_cost, total_monetary_cost], dim=-1), None, [total_query_cost, total_monetary_cost]
        else:
            raise NotImplementedError("Currently only supports 2 objectives")

    @staticmethod
    def make_dataset(*args, **kwargs):
        return BlockDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateBlockSelection.initialize(*args, **kwargs)



