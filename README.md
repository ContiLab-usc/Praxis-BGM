# Praxis-BGM

## Prior-Augmented Bayesian Gaussian Mixture Model via Natural-Gradient Variational Inference

Praxis-BGM is a **semi-supervised transfer-learning framework** for clustering high-dimensional omics, multi-omics, and single-cell data using **Bayesian Gaussian Mixture Models** with **Natural-Gradient Variational Inference (NGVI)**.

This method enables incorporation of **cluster-specific prior information**—including means, covariances, sparsity masks, and mixing weights—from a labeled *source dataset* to guide clustering in an unlabeled *target dataset*. Praxis-BGM is implemented in **JAX**, providing GPU/TPU acceleration and numerically stable updates.

Praxis-BGM corresponds to the method described in:

> **Qiran Jia, Jesse A. Goodrich, David V. Conti**  
> *Clustering of Omic Data Using Semi-Supervised Transfer Learning for Gaussian Mixture Models via Natural-Gradient Variational Inference*.  
> bioRxiv 2025.11.13.688299v2.

---

## Features

- Semi-supervised Bayesian transfer learning for Gaussian mixture models  
- NGVI (VON algorithm) for fast, stable optimization  
- Four classes of priors:
  - Means `μ` for each cluster
  - Covariances `Σ` for each cluster (optional)
  - Structural adjacency masks `A` derived from pathway databases (optional)
  - Mixing weights `θ` (optional)
- Structural sparsity masks `A` to encode pathway or network knowledge  
- Mini-batch training, efficient with n >> 10,000
- Bayes factor–based feature importance scoring  
- Compatible with high-dimensional omics (d > 1,000)

---

## Installation

Clone the repository and install locally:

```bash
pip install -e .
```
or install directly from this repo

```bash
pip install git+https://github.com/ContiLab-usc/Praxis-BGM.git
```

## Requirements
python >= 3.9
jax >= 0.4.20
jaxlib >= 0.4.20
numpy
scikit-learn
matplotlib

For GPU acceleration, install the appropriate jaxlib from:
https://github.com/google/jax#installation

## Tutorial

A complete walkthrough of Praxis-BGM is provided in the following notebook:

[Praxis_BGM_Tutorial.ipynb](./Praxis_BGM_Tutorial.ipynb)

This tutorial demonstrates the basic workflow, including simulation of overlapping GMM data, construction of source → target domain shift, empirical source priors, NGVI-based clustering with transferred priors, and benchmarking against QDA and unsupervised baselines using ARI.

## Citation

If you use Praxis-BGM in your research, please cite:

@article{jia2025praxisbgm,
  title={Clustering of Omic Data Using Semi-Supervised Transfer Learning for Gaussian Mixture Models via Natural-Gradient Variational Inference},
  author={Jia, Qiran and Goodrich, Jesse A. and Conti, David V.},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.11.13.688299},
}
