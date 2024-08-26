# misty-rainforest
A wrapper to benchmark execution time and memory usage of various quantum benchmarking suites.

# Installation
- Clone this repository using the command
```bash
git clone <repo_link>
```  

- Create a conda environment by using the YAML file provided
```bash
conda env create -f environment.yml
```
A few libraries may be missing, in which case it is recommended to install them via pip.

# Code execution  

Get the plots for the metrics of the circuits:
```bash
python3 compute_metrics.py
```

