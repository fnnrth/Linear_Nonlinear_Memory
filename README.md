# Almost Linear RNNs for Memory Tasks

A collection of memory tasks implemented using AL-RNNs from the paper "Uncovering the Functional Roles of Nonlinearity in Memory".

## Project Structure

Each task is organized in its own subdirectory:

- **Classification Problems/**: Various classification tasks with memory requirements
- **Copy Problem/**: Sequence copying task with variable delays
- **Context Switching/**: Context-dependent evidence integration task
- **Addition Problem/**: Simple addition task with context-dependent computation
- **SCAN task/**: Language-like instruction following task

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
python context_switching/memory_launcher.py --M 2 --L 2

# Run multiple experiments in parallel
python classification_problems/launcher_mnist.py
```

For multiple experiments, you can select different combinations of hyperparameters, number of processes executed simultaneously, and number of threads used.
See individual task directories for specific hyperparameters and usage details.
