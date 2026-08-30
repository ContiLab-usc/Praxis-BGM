# Praxis-BGM
High-dimensional omic data are typically measured on limited sample sizes, which challenges model-based
clustering methods such as Gaussian mixture models, often leading to instability and poor generalization under complex
mixture structures. To address these limitations, we developed Praxis-BGM, a natural-gradient variational inference
framework for Gaussian mixture models that incorporates informative priors—cluster-specific means, covariances, and
structural connectivity—from large-scale reference data with robust cluster structures to enable semi-supervised transfer
learning on a small target dataset. 
![Praxis-BGM overview](./visual_abstract.png)
## Updated installable layout for the latest Praxis algorithm

This folder packages the newest `Praxis` algorithm in the installable
`Praxis-BGM` repository structure. The core implementation now lives under
`src/praxis_bgm/`, while tutorials and runnable examples stay at the repository
root.

The main user-facing class is:

```python
from praxis_bgm import Praxis_BGM
```

## What's included

- `src/praxis_bgm/core.py`: high-level `Praxis_BGM` model API
- `src/praxis_bgm/utility.py`: numerical helpers and damped NGVI updates
- `src/praxis_bgm/prior_utils.py`: source-to-target alignment and prior builders
- `Praxis_tutorial.py`: end-to-end transfer-learning tutorial script
- `Praxis_BGM_Tutorial.ipynb`: notebook walkthrough

### Recommended environment
To keep the package and tutorials in one place, use a
dedicated Conda environment named `Praxis_env`.

Create and activate the environment:

```bash
conda create -n Praxis_env python=3.10 -y
conda activate Praxis_env
```

Core libraries required for the packaged `praxis_bgm` library and the main
Python tutorial:

- `jax`
- `jaxlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `scipy`
- `statsmodels`
  
## Installation
From the repository root:

```bash
pip install -e .
```

Or install directly from a local checkout:

```bash
pip install git+https://github.com/ContiLab-usc/Praxis-BGM.git
```

## Repository layout

```text
Praxis/
├── LICENSE
├── README.md
├── Praxis_BGM_Tutorial.ipynb
├── Praxis_tutorial.py
├── pyproject.toml
└── src/
    └── praxis_bgm/
        ├── __init__.py
        ├── core.py
        ├── prior_utils.py
        └── utility.py
```

## Quick start

```python
import jax
import numpy as np
from praxis_bgm import Praxis_BGM

X = np.array(
    [
        [-2.2, -1.8],
        [-1.8, -2.3],
        [1.7, 2.5],
        [2.3, 1.8],
    ],
    dtype=np.float32,
)

prior_mus = np.array(
    [
        [-2.0, -2.0],
        [2.0, 2.0],
    ],
    dtype=np.float32,
)

model = Praxis_BGM(
    rng_key=jax.random.PRNGKey(0),
    K=2,
    prior_mus=prior_mus,
    beta=1e-3,
    num_samples=8,
    elbo_eval_freq=1,
    verbose=True,
)

model.fit(X, num_iters=200, batch_size=4)

assignments, weights = model.predict(X)
posterior_mus, posterior_covs, posterior_pis, responsibilities = model.get_posteriors(X)

print("Assignments:", assignments)
print("Weights:", weights)
print("Posterior means:\n", posterior_mus)
print("Posterior covariances:\n", posterior_covs)
print("Posterior mixture weights:", posterior_pis)
print("Responsibilities:\n", responsibilities)
```

## Main model arguments

- `K`: number of mixture components
- `prior_mus`, `prior_Sigmas`, `prior_weights`: optional transferred priors
- `init_mus`, `init_covs`, `init_pis`: optional variational initialization
- `beta`: step size for mean and weight updates
- `num_samples`: number of anchor samples used per NGVI update
- `sparse_A` / `cluster_A`: optional structural masks


## Tutorials

- Script walkthrough: [Praxis_tutorial.py](./Praxis_tutorial.py)
- Notebook walkthrough: [Praxis_BGM_Tutorial.ipynb](./Praxis_BGM_Tutorial.ipynb)


## R interface

An R wrapper is maintained in a separate repository for users who prefer to run
Praxis-BGM from R:

**https://github.com/BruceResearch/Praxis_BGM_R_interface**

The wrapper does not reimplement the model. It calls this Python/JAX package
directly through [`reticulate`](https://rstudio.github.io/reticulate/), so you
still need a working Python install of `praxis_bgm` (see
[Installation](#installation) above). All wrapper logic lives in a single
source-of-truth script, `R/praxis_bgm_interface.R`, which exposes
`praxis_bgm_fit()`, `praxis_bgm_bf_selection()`, and the simulation helpers used
by its tutorials. Its argument names track the current Python API
(`prior_weights`, `init_pis`, `freeze_A_zeros`, `data_precision_int`,
`likelihood_temp`, `rho_prec`, `rho_mu`, `elbo_eval_freq`, ...), and results are
returned with both R-friendly (1-based) and raw Python (0-based) indexing.

### Setup

1. Create the Python environment and install this package:

   ```bash
   conda create -n Praxis_env python=3.10 -y
   conda activate Praxis_env
   pip install jax jaxlib numpy scikit-learn matplotlib
   pip install git+https://github.com/ContiLab-usc/Praxis-BGM.git
   ```

2. Install the required R packages: `reticulate`, `MASS`, `proxy`, `clue`
   (`mclust` and `rmarkdown` are suggested for the tutorials).

3. Clone the R interface repo, point `reticulate` at the environment, and source
   the wrapper script:

   ```r
   library(reticulate)

   praxis_python <- Sys.getenv("RETICULATE_PYTHON", unset = "")
   if (nzchar(praxis_python)) {
     use_python(praxis_python, required = TRUE)
   } else {
     use_condaenv("Praxis_env", required = TRUE)
   }

   source("R/praxis_bgm_interface.R")
   ```

### Basic usage

```r
fit <- praxis_bgm_fit(
  data = your_matrix,
  K = 3,
  seed = 123,
  prior_weights = c(1/3, 1/3, 1/3),
  num_iters = 50,
  batch_size = min(50, nrow(your_matrix)),
  verbose = FALSE
)
# fit$assignments (1-based), fit$assignments_zero_based, fit$learned_weights,
# fit$posterior_mus / posterior_covs / posterior_pis, fit$elbo_history, fit$model

bf <- praxis_bgm_bf_selection(
  model = fit,
  data = your_matrix,
  top_n = 20,
  visual = FALSE
)
# bf$top_features (1-based), bf$top_features_zero_based, bf$classification
```

See the R interface repository's README, `Praxis_R_Wrapper.Rmd` (compact
example), and `Praxis_R_Tutorial.Rmd` (end-to-end walkthrough with transferred
priors) for the full argument reference. A copy of the wrapper walkthrough is
also included here as [Praxis_R_Wrapper.Rmd](./Praxis_R_Wrapper.Rmd).

## Notes

- The package requires Python 3.9+.
- JAX is used for the core optimization and Monte Carlo ELBO evaluation.

## Citation

If you use Praxis-BGM in your research, please cite:

> **Qiran Jia, Jesse A. Goodrich, David V. Conti.**
> *Praxis-BGM: clustering of omics data using semi-supervised transfer learning for Gaussian mixture models via natural-gradient variational inference.*
> Bioinformatics, 42(6):btag395, 2026. doi:10.1093/bioinformatics/btag395

```bibtex
@article{jia2026praxisbgm,
  title={Praxis-BGM: clustering of omics data using semi-supervised transfer learning for Gaussian mixture models via natural-gradient variational inference},
  author={Jia, Qiran and Goodrich, Jesse A. and Conti, David V.},
  journal={Bioinformatics},
  volume={42},
  number={6},
  pages={btag395},
  year={2026},
  doi={10.1093/bioinformatics/btag395}
}
```

