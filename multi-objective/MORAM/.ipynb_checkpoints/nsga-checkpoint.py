import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3

from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.operators.sampling.rnd import FloatRandomSampling, BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation


class BlockchainStorageOptimization(ElementwiseProblem):
    def __init__(self, N, query_costs, query_freqs, sizes, u_s, u_r, u_t, D_L):
        
        super().__init__(n_var=N, n_obj=2, n_ieq_constr=1, xl=0, xu=1)
        self.query_costs = np.array(query_costs)
        self.sizes = np.array(sizes)
        self.query_freqs = np.array(query_freqs)
        self.u_s = u_s
        self.u_r = u_r
        self.u_t = u_t
        self.D_L = D_L

    def _evaluate(self, x, out, *args, **kwargs):
        # Ensure x is a numpy array for element-wise operations
        x = np.array(x)
        
        C_Q = np.sum(x * self.query_costs)

        local_query_freqs = np.sum(x * self.query_freqs)  # Sum of query frequencies for blocks in local storage
        C_M = (self.u_s * sum(x * self.sizes)) + (self.u_r * local_query_freqs) + (self.u_t * sum(x * self.query_freqs * self.sizes))
        
        # Calculate the usage of local storage
        local_storage_usage = sum(x * self.sizes)
        
        # Objectives (negated for maximization)
        out["F"] = -np.array([C_Q, C_M])
        
        # Constraints (should be of the form g_i(x) <= 0)
        out["G"] = np.array([local_storage_usage - self.D_L])



def nsga2bsp(target_blocks, max_capacity=100.0):

    # Make sure to replace these with your actual problem parameters
    N = len(target_blocks) # Example: Number of blocks
    query_costs = np.array(list(block.query_cost for block in target_blocks))  # Example: Query costs for each block
    sizes = np.array(list(block.size for block in target_blocks))  # Example: Sizes of each block
    query_freqs = np.array(list(block.query_freq for block in target_blocks))
    u_s = 0.000027  # Unit monetary cost for storage
    u_r = 0.000006  # Unit monetary cost for number of requests
    u_t = 0.00015  # Unit monetary cost for total transmission size
    D_L = max_capacity  # Maximum local space occupancy ratio

    problem = BlockchainStorageOptimization(N, query_costs, query_freqs, sizes, u_s, u_r, u_t, D_L)
    
    algorithm = NSGA2(
        pop_size=200,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=0.8),
        mutation=BitflipMutation(prob=0.3, prob_var=0.3),
        eliminate_duplicates=True
    )

    result = minimize(problem,
                      algorithm,
                      termination=('n_gen', 100),
                      verbose=False)

    if result.X is None:
        print("No solutions were found.")
        return None, None

    
    total_block_sizes = np.sum(result.X * sizes, axis=1)  # Multiply x by sizes and sum along axis 1 (per solution)
    
    # Return all solutions found
    return -1 * result.F, total_block_sizes



def nsga3bsp(target_blocks, max_capacity=100.0):

    # Make sure to replace these with your actual problem parameters
    N = len(target_blocks) # Example: Number of blocks
    query_costs = np.array(list(block.query_cost for block in target_blocks))  # Example: Query costs for each block
    sizes = np.array(list(block.size for block in target_blocks))  # Example: Sizes of each block
    query_freqs = np.array(list(block.query_freq for block in target_blocks))
    u_s = 0.000027  # Unit monetary cost for storage
    u_r = 0.000006  # Unit monetary cost for number of requests
    u_t = 0.00015  # Unit monetary cost for total transmission size
    D_L = max_capacity  # Maximum local space occupancy ratio

    problem = BlockchainStorageOptimization(N, query_costs, query_freqs, sizes, u_s, u_r, u_t, D_L)

    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
    
    algorithm = NSGA3(
        pop_size=200,
        ref_dirs=ref_dirs,
        sampling=BinaryRandomSampling(),
        crossover=TwoPointCrossover(prob=0.8),
        mutation=BitflipMutation(prob=0.3, prob_var=0.3),
        eliminate_duplicates=True
    )

    result = minimize(problem,
                      algorithm,
                      termination=('n_gen', 100),
                      verbose=False)

    if result.X is None:
        print("No solutions were found.")
        return None, None
    
    total_block_sizes = np.sum(result.X * sizes, axis=1)  # Multiply x by sizes and sum along axis 1 (per solution)
    
    # Return all solutions found
    return -1 * result.F, total_block_sizes