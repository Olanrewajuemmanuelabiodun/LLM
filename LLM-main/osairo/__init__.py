"""
osairo: A flexible, all-purpose molecular modeling, quantum chemistry, and
machine learning Active Learning platform, orchestrated with an LLM.

Modules:
- data_manager: Functions to handle CSV and data management
- model_manager: Functions to train/handle predictive models (GP, NN, etc.)
- active_learning: Core AL loop (uncertainty estimation, re-training, etc.)
- simulation_scripts: Generators for molecular/quantum simulation input scripts
- job_scripts: Generators for job submission scripts (UGE, Slurm, etc.)
- cli: Command-line interface for interactive usage
- config: Global config & environment variable handling
"""
__version__ = "0.1.0"
