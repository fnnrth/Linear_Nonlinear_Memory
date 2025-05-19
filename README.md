# Almost Linear RNNs for Memory Tasks

A collection of memory tasks implemented using RNNs with linear and nonlinear components. Each task explores different aspects of memory and computation in recurrent neural networks.

## Project Structure

Each task is organized in its own directory with a consistent structure:

- **Addition Problem/**: Simple addition task with context-dependent computation
- **Copy Problem/**: Sequence copying task with variable delays
- **SCAN task/**: Language-like instruction following task
- **Classification Problems/**: Various classification tasks with memory requirements
- **Context Switching/**: Context-dependent evidence integration task

### Task Directory Structure

Each task directory contains:
- `load_data.py`: Dataset generation and data loading utilities
- `*_task.py`: Core model architecture and training/testing functions
- `*_launcher.py`: Script for running experiments

### Shared Utilities

- `MAR.py`: Manifold Attractor Regularization implementation
- `analysis_functions.py`: Functions bitcode representation and visualization

## Usage

Each task can be run independently using its launcher script. For example:
```bash
# Run a single experiment
python Context_Switching/memory_launcher.py --M 2 --L 2

# Run multiple experiments in parallel
python Classification Problems/launcher_mnist.py
```

See individual task directories for specific hyperparameters and usage details.
