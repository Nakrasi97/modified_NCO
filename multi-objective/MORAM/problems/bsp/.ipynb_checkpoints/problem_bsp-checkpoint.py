from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
from problems.bsp.state_bsp import StateBlockSelection
import simpy
import numpy as np
from tqdm import tqdm

class Block:
    def __init__(self, index, size):
        self.index = index
        self.size = size
        self.query_cost = 0.0
        self.query_freq = 0
        self.rec_access = 0.0

class BlockDataset(Dataset):
    def __init__(self, num_blocks, num_samples, min_size=0.1, max_size=2.0, mean_size=1.0, std_dev=0.25, duration=0.5, batch_size=10):
        super(BlockDataset, self).__init__()

        self.num_samples = num_samples
        self.num_blocks = num_blocks
        self.batch_size = batch_size
        self.duration = duration

        # Initialize the dataset with a progress bar
        self.data = []
        for _ in tqdm(range(num_samples), desc="Creating dataset samples"):
            blocks = self.generate_blocks(num_blocks, min_size, max_size, mean_size, std_dev)
            blocks = self.simulate_queries_with_simpy(blocks)
            block_data = self.extract_block_data(blocks)
            self.data.append(block_data)

        self.size = len(self.data)

    def generate_blocks(self, num_blocks, min_size, max_size, mean_size, std_dev):
        blocks = []
        for index in range(num_blocks):
            size = np.clip(np.random.normal(mean_size, std_dev), min_size, max_size)
            blocks.append(Block(index, size))
        return blocks

    def simulate_queries_with_simpy(self, blocks):
        conn_bandwidth = 100  # in Mbps
        delay = 0.02  # seconds

        indices = np.arange(len(blocks))
        mean = len(indices) / 2
        std_dev = len(indices) / 6

        # Prepare NumPy arrays for efficient updates
        block_sizes = np.array([block.size for block in blocks])
        query_costs = np.zeros(len(blocks))
        query_freqs = np.zeros(len(blocks))

        def normdist_query_gen(env, duration, batch_size):
            while env.now < duration * 60:
                # Generate a batch of normally distributed indices
                norm_indices = np.random.normal(mean, std_dev, batch_size).astype(int)
                norm_indices = np.clip(norm_indices, 0, len(indices) - 1)
                
                # Vectorized update of the query costs and frequencies
                for qblock_index in norm_indices:
                    query_costs[qblock_index] += (block_sizes[qblock_index] * 8) / conn_bandwidth
                    query_freqs[qblock_index] += 1
                
                # Wait for the next batch to be processed
                yield env.timeout(delay * batch_size)
            
            return query_costs, query_freqs

        env = simpy.Environment()
        env.process(normdist_query_gen(env, self.duration, self.batch_size))
        env.run(until=self.duration * 60)

        # Assign the results back to the blocks
        for i, block in enumerate(blocks):
            block.query_cost = query_costs[i]
            block.query_freq = query_freqs[i]

        return blocks

    def extract_block_data(self, blocks):
        block_data = []
        for block in blocks:
            block_info = [block.query_cost, block.size, block.query_freq, block.rec_access]
            block_data.append(block_info)
        return torch.FloatTensor(block_data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]





class BSP(object):
    NAME = 'block_selection'

    @staticmethod
    def get_costs(dataset, pi, w, num_objs, max_cap, method='weighted_sum'):
        # Ensure the selection is a valid permutation of 0s and 1s
        assert (
            (pi == 0) | (pi == 1)
        ).all(), "Invalid block selection - must be 0s and 1s only"
        
        # Ensure the sum of 1s (selected blocks) does not exceed the number of blocks
        num_blocks = pi.size(1)
        assert (
            pi.sum(1) <= num_blocks
        ).all(), "Invalid block selection - selection sum exceeds number of blocks"
    
        # Filter dataset to only include blocks that were selected (where pi == 1)
        d = dataset * pi.unsqueeze(-1).expand_as(dataset)
    
        # Assume first column is query cost, second is block size, third is request frequency, fourth is transmission count
        query_cost = d[..., 0]
        block_size = d[..., 1]
        request_freq = d[..., 2]
        transmission_count = d[..., 3]
    
        # Calculate the total block size for selected blocks
        total_block_size = block_size.sum(1)
    
        # Assert that the total block size does not exceed the maximum capacity
        assert (total_block_size <= max_cap).all(), "Stored blocks exceed storage capacity"
    
        # Calculate the total query cost for selected blocks
        total_query_cost = query_cost.sum(1)
        
        # Calculate the total monetary cost for storing the selected blocks
        storage_cost = w[:, 0].unsqueeze(1) * total_block_size
        request_cost = w[:, 1].unsqueeze(1) * request_freq.sum(1)
        transmission_cost = w[:, 2].unsqueeze(1) * (block_size * transmission_count).sum(1)
        total_monetary_cost = storage_cost + request_cost + transmission_cost
    
        # Stack the objectives
        stacked_costs = torch.stack([total_query_cost, total_monetary_cost], dim=-1)
    
        if method == 'weighted_sum':
            # Weighted sum
            weighted_sum = (w.unsqueeze(1) * stacked_costs).sum(-1).detach()
            return weighted_sum, None, [total_query_cost, total_monetary_cost]
    
        elif method == 'tchebycheff':
            # Tchebycheff scalarization
            w_rep = w.unsqueeze(0).expand(stacked_costs.size(0), -1, -1)
            tchebycheff = (w_rep * stacked_costs.unsqueeze(1)).max(-1)[0].detach()
            return tchebycheff, None, [total_query_cost, total_monetary_cost]
    
        else:
            raise ValueError("Unknown method: {}. Use 'weighted_sum' or 'tchebycheff'.".format(method))


    @staticmethod
    def make_dataset(*args, **kwargs):
        return BlockDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateBlockSelection.initialize(*args, **kwargs)





