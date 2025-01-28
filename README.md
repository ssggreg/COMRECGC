# COMRECGC: Global Graph Counterfactual Explainer through Common Recourse

This repository is a reference implementation of the Common Recourse explainer from the paper:
<br/>
> COMRECGC: Global Graph Counterfactual Explainer through Common
Recourse.<br>

## Requirements

We tested our code in Python 3.10.12 using the following main dependencies:

- [PyTorch](https://pytorch.org/get-started/locally/) v1.8.0
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) v2.3.1
- [NetworkX](https://networkx.org/documentation/networkx-2.5/install.html) v3.3
- [NumPY](https://numpy.org/install/) v1.26.4
- [tqdm](https://tqdm.github.io/) v4.66.5
- [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) v2024.3.1

W ran our experiments on the Google Colab platform, using an L4 NVIDIA GPU (22.5GB of RAM).

## GNN and distance evaluation

We provide gnn and neurosed base models in the repository. To train new models:

- For gnn base models, the model we use are in the [gnn.py](gnn.py) module.
- For neurosed base models, please refer to the GREED [neurosed](https://github.com/idea-iitd/greed) repository.

## Experiments
To run the experiments from the paper, please run the COMRECGC notebook.

