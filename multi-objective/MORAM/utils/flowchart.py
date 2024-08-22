from graphviz import Digraph
import os

# Set the PATH to include Graphviz
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

def create_rl_training_flowchart():
    # Create a new directed graph
    dot = Digraph(comment='MORAM Model Training Process with Policy Gradient')

    # Start and Initialization
    dot.node('A', 'Start Training', shape='ellipse')
    dot.node('B', 'Initialize Options and Model', shape='box')

    # Training Loop
    dot.node('C', 'Training Loop (Epochs)', shape='box')
    dot.node('D', 'Generate Training Data', shape='box')
    dot.node('E', 'Set Model to Training Mode', shape='box')

    # Train Epoch
    dot.node('F', 'Train Epoch (Batches)', shape='box')

    # Train Batch
    dot.node('G', 'Generate Sequence (Policy)', shape='box')
    dot.node('H', 'Evaluate Sequence (Cost Calculation)', shape='box')
    dot.node('I', 'Compute Log Likelihood', shape='box')
    dot.node('J', 'Compute Reinforce Loss', shape='box')
    dot.node('K', 'Compare to Baseline', shape='box')
    dot.node('L', 'Update Model Parameters (Backpropagation)', shape='box')

    # Logging and Validation
    dot.node('M', 'Logging', shape='box')
    dot.node('N', 'Validation', shape='box')
    dot.node('O', 'Save Checkpoint', shape='box')

    # End Training
    dot.node('P', 'End Training', shape='ellipse')

    # Connect the nodes
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F')
    dot.edge('F', 'G')
    dot.edge('G', 'H')
    dot.edge('H', 'I')
    dot.edge('I', 'J')
    dot.edge('J', 'K')
    dot.edge('K', 'L')
    dot.edge('L', 'M')
    dot.edge('M', 'F', label='Next Batch')
    dot.edge('F', 'N', label='End of Epoch')
    dot.edge('N', 'O')
    dot.edge('O', 'C', label='Next Epoch')
    dot.edge('O', 'P')

    # Render the flowchart to a file
    dot.render('rl_training_flowchart.gv', view=True, format='png')

# Create and render the flowchart
create_rl_training_flowchart()
