# COMRECGC: Global Graph Counterfactual Explainer through Common Recourse

This repository is a reference implementation of the Common Recourse explainer from the paper:

> **COMRECGC: Global Graph Counterfactual Explainer through Common Recourse**  
> Gregoire Fournier, Sourav Medya  
> [arXiv:2505.07081](https://arxiv.org/abs/2505.07081)


📌 This repository contains the official implementation of the method by the first author, **Gregoire Fournier**.

## Requirements

We tested our code in Python 3.10.12 using the following main dependencies:

- [PyTorch](https://pytorch.org/get-started/locally/) v1.8.0
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) v2.3.1
- [NetworkX](https://networkx.org/documentation/networkx-2.5/install.html) v3.3
- [NumPY](https://numpy.org/install/) v1.26.4
- [tqdm](https://tqdm.github.io/) v4.66.5
- [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) v2024.3.1

We ran our experiments on the Google Colab platform, using an L4 NVIDIA GPU (22.5GB of RAM).

## GNN and distance evaluation

We provide gnn and neurosed base models in the repository. To train new models:

- For gnn base models, the model we use are in the [gnn.py](gnn.py) module.
- For neurosed base models, please refer to the GREED [neurosed](https://github.com/idea-iitd/greed) repository.

## Experiments

To reproduce the experiments from the paper, run the following shell script:

```bash
bash run_experiments.sh
```
This script runs the full suite of model training and evaluation across all datasets.
Interactive Analysis

For interactive exploration, visualizations, or debugging, you can use the provided Jupyter notebook:

    comrecgc.ipynb: Includes tables and plots of the results generated by the experiments.

Note: Make sure to run the experiments first, as the notebook assumes the output files have already been generated.
