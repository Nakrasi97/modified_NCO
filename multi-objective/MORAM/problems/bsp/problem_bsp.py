from torch.utils.data import Dataset
import torch
from problems.bsp.state_bsp import StateBlockSelection
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
    def get_costs(dataset, pi, w, num_objs, max_cap):
        # Ensure the selection is a valid permutation of 0s and 1s
        assert (
            (pi == 0) | (pi == 1)
        ).all(), "Invalid block selection - must be 0s and 1s only"
        
        # Ensure the sum of 1s (selected blocks) does not exceed the number of blocks
        num_blocks = pi.size(1)
        assert (
            pi.sum(1) <= num_blocks
        ).all(), "Invalid block selection - selection sum exceeds number of blocks"

        # Filter dataset to only include blocks that were not selected (where pi == 0)
        d = dataset * (1 - pi).unsqueeze(-1).expand_as(dataset)

        # Assume first column is query cost, second is block size, third is request frequency, fourth is transmission count
        query_cost = d[..., 0]
        block_size = d[..., 1]
        request_freq = d[..., 2]
        transmission_count = d[..., 3]

        # Calculate the total block size for non-selected blocks
        total_block_size = block_size.sum(1)

        # Assert that the total block size does not exceed the maximum capacity
        assert (total_block_size <= max_cap).all(), "Stored blocks exceed storage capacity"

        # Calculate the total query cost for non-selected blocks
        total_query_cost = query_cost.sum(1)
        
        # Calculate the total monetary cost for storing the non-selected blocks
        storage_cost = w[:, 0].unsqueeze(1) * block_size.sum(1)
        request_cost = w[:, 1].unsqueeze(1) * request_freq.sum(1)
        transmission_cost = w[:, 2].unsqueeze(1) * (block_size * transmission_count).sum(1)
        total_monetary_cost = storage_cost + request_cost + transmission_cost

        if num_objs == 2:
            return torch.stack([total_query_cost, total_monetary_cost], dim=-1), None, [total_query_cost, total_monetary_cost]
        else:
            raise NotImplementedError("Currently only supports 2 objectives")





