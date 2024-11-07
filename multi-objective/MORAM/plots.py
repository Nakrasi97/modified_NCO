import matplotlib.pyplot as plt
import numpy as np
import torch
import time

def plot_comparisons(results):
    """
    Generate bar graphs comparing average costs, local storage, and runtime between the model and NSGA algorithms.
    """

    algorithms = ['model', 'nsga2', 'nsga3']
    
    # Compute averages
    query_cost_means = [np.mean(results[alg]['query_costs']) for alg in algorithms]
    monetary_cost_means = [np.mean(results[alg]['monetary_costs']) for alg in algorithms]
    storage_means = [np.mean(results[alg]['local_storage']) for alg in algorithms]
    runtime_means = [results[alg]['runtime'] for alg in algorithms]
    
    # Plot for query costs
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, query_cost_means, color=['blue', 'green', 'orange'])
    for i, v in enumerate(query_cost_means):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    plt.title('Average Query Costs Comparison')
    plt.xlabel('Algorithms')
    plt.ylabel('Average Query Cost')
    plt.savefig(f'./graphs/query_cost_comparison_{time.time()}.png')
    plt.close()

    # Plot for monetary costs
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, monetary_cost_means, color=['blue', 'green', 'orange'])
    for i, v in enumerate(monetary_cost_means):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    plt.title('Average Monetary Costs Comparison')
    plt.xlabel('Algorithms')
    plt.ylabel('Average Monetary Cost')
    plt.savefig(f'./graphs/monetary_cost_comparison_{time.time()}.png')
    plt.close()

    # Plot for local storage used
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, storage_means, color=['blue', 'green', 'orange'])
    for i, v in enumerate(storage_means):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    plt.title('Average Local Storage Used Comparison')
    plt.xlabel('Algorithms')
    plt.ylabel('Average Local Storage Used')
    plt.savefig(f'./graphs/storage_comparison_{time.time()}.png')
    plt.close()

    # Plot for runtime
    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, runtime_means, color=['blue', 'green', 'orange'])
    for i, v in enumerate(runtime_means):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    plt.title('Runtime Comparison')
    plt.xlabel('Algorithms')
    plt.ylabel('Runtime (seconds)')
    plt.savefig(f'./graphs/runtime_comparison_{time.time()}.png')
    plt.close()


def plot_pareto_subplots(nds_list_model, nds_list_nsga2, nds_list_nsga3, title, save_path):
    """
    Plots the Pareto fronts for multiple samples as subplots in one image, comparing between model, NSGA2, and NSGA3.
    
    :param nds_list_model: List of Pareto fronts (NDS) for the model's predictions.
    :param nds_list_nsga2: List of Pareto fronts (NDS) for NSGA2 solutions.
    :param nds_list_nsga3: List of Pareto fronts (NDS) for NSGA3 solutions.
    :param title: Title for the overall plot.
    :param save_path: File path to save the plot.
    """
    num_samples = len(nds_list_model)
    cols = 2  # You can set how many columns you want in the subplots
    rows = (num_samples + cols - 1) // cols  # Number of rows required
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    fig.suptitle(title)

    for i in range(num_samples):
        ax = axes[i // cols, i % cols]
        
        # Plot Model's NDS
        pareto_points_model = nds_list_model[i].cpu().numpy()
        ax.scatter(pareto_points_model[:, 0], pareto_points_model[:, 1], color='blue', label="Model NDS")

        # Plot NSGA2's NDS
        pareto_points_nsga2 = np.array(nds_list_nsga2[i])
        ax.scatter(pareto_points_nsga2[:, 0], pareto_points_nsga2[:, 1], color='green', label="NSGA2 NDS")

        # Plot NSGA3's NDS
        pareto_points_nsga3 = np.array(nds_list_nsga3[i])
        ax.scatter(pareto_points_nsga3[:, 0], pareto_points_nsga3[:, 1], color='black', label="NSGA3 NDS")

        ax.set_title(f'Sample {i+1}')
        ax.set_xlabel('Total Query Cost')
        ax.set_ylabel('Total Monetary Savings')
        ax.legend()
        ax.grid(True)
    
    # Remove empty subplots
    if num_samples < rows * cols:
        for j in range(num_samples, rows * cols):
            fig.delaxes(axes.flatten()[j])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for title
    plt.savefig(save_path)
    plt.close()
