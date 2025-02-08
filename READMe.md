# osairo

**osairo** is an all-purpose scientific software tool for molecular modeling, quantum chemistry, machine learning active learning, and HPC job script generation. It features:

- **Data management:** Load CSV files and specify input/target features.
- **Model training:** Train a Gaussian Process (via GPFlow) or a Neural Network (via TensorFlow).
- **Active Learning:** Identify the most uncertain data points and automatically generate simulation scripts.
- **HPC Job Submission:** Generate job scripts for UGE or Slurm based on your simulation inputs.
- **Interactive Chat Mode:** Ask questions and get answers using an LLM-powered knowledge assistant.

- **Make sure to provide your API key in the config.py file**

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/theOsaroJ/osairo-scientist-llm.git
cd osairo
pip install -e .
pip install -U langchain-openai click pandas numpy scikit-learn
