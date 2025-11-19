#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tutorial_transfer_priors.py

Demo: Transfer learned cluster means & covariances from a large source dataset
      to a smaller target dataset with global + feature-specific shifts.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split
from sklearn.metrics import adjusted_rand_score

# Import the Praxis_BGM model from the local package folder
# Make sure the folder is named `praxis_bgm` and contains __init__.py
from Praxis_BGM import Praxis_BGM


# ---------------------------------------------------------------------
# 1) Helpers: simulate GMM data (with controllable covariance structure)
# ---------------------------------------------------------------------
def make_block_cov(d, block=10, on_diag_var=1.0, off_diag=0.3, jitter=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    Sigma = np.eye(d) * on_diag_var
    for start in range(0, d, block):
        end = min(start + block, d)
        if end - start >= 2:
            block_idx = slice(start, end)
            Sigma[block_idx, block_idx] += off_diag * (
                np.ones((end - start, end - start)) - np.eye(end - start)
            )
    Sigma += jitter * np.eye(d)
    return Sigma


def simulate_gmm(N, K, d, seed=0, pis=None):
    """
    Simulate a K-component GMM in d-dim with diverse covariances and separated means.
    If `pis` is provided, it must be a length-K probability vector.
    Returns X (N,d), z (N,), mus (K,d), Sigmas (K,d,d), pis (K,)
    """
    rng = np.random.default_rng(seed)
    base_dirs = rng.normal(size=(K, d))
    base_dirs /= np.linalg.norm(base_dirs, axis=1, keepdims=True) + 1e-12
    mus = 1 * base_dirs
    Sigmas = np.stack(
        [
            make_block_cov(
                d,
                block=8,
                on_diag_var=1.0 + 0.3 * k,
                off_diag=0.25,
                seed=seed + 10 * k,
            )
            for k in range(K)
        ],
        axis=0,
    )

    if pis is None:
        pis = rng.dirichlet(alpha=np.ones(K))
    else:
        pis = np.asarray(pis, dtype=np.float32)
        pis = pis / pis.sum()

    z = rng.choice(K, size=N, p=pis)
    X = np.zeros((N, d), dtype=np.float32)
    for k in range(K):
        nk = np.sum(z == k)
        if nk > 0:
            X[z == k] = rng.multivariate_normal(mus[k], Sigmas[k], size=nk)
    return (
        X.astype(np.float32),
        z.astype(int),
        mus.astype(np.float32),
        Sigmas.astype(np.float32),
        pis.astype(np.float32),
    )


# ---------------------------------------------------------------------
# 2) Build a shifted "target" from the source parameters
# ---------------------------------------------------------------------
def make_target_params_from_source(
    src_mus,
    src_Sigmas,
    frac_feat_shift=0.2,
    global_shift_scale=0.6,
    feat_shift_scale=0.6,
    cov_scale=1.1,
    flip_frac=0.15,  # fraction of features to polarity-flip (reversed causal)
    flip_per_cluster=True,  # flip signs per cluster for selected features
    seed=123,
):
    """
    Start from source (mus, Sigmas),
    - add global shift
    - add feature-specific shifts on a fraction of features
    - inflate covariances a bit
    - optionally flip the *sign* of selected features (per cluster) to simulate reversed causality.
    """
    src_mus = np.asarray(src_mus, dtype=np.float32)
    src_Sigmas = np.asarray(src_Sigmas, dtype=np.float32)
    rng = np.random.default_rng(seed)
    K, d = src_mus.shape

    # Global shift
    g = rng.normal(scale=global_shift_scale, size=(d,)).astype(np.float32)

    # Feature-specific shifts
    m = max(1, int(np.floor(frac_feat_shift * d)))
    feat_idx = rng.choice(np.arange(d), size=m, replace=False)
    E = np.zeros((K, d), dtype=np.float32)
    E[:, feat_idx] = rng.normal(scale=feat_shift_scale, size=(K, m)).astype(np.float32)

    tgt_mus = src_mus + g[None, :] + E

    # Optional polarity flips (reversed causal features)
    m_flip = max(1, int(np.floor(flip_frac * d)))
    flip_idx = rng.choice(np.arange(d), size=m_flip, replace=False)
    if flip_per_cluster:
        # random {+1,-1} per cluster for those features
        signs = rng.choice(
            np.array([-1.0, 1.0], dtype=np.float32), size=(K,), replace=True
        )
        tgt_mus[:, flip_idx] = tgt_mus[:, flip_idx] * signs[:, None]
    else:
        # flip the same way for all clusters on selected features
        tgt_mus[:, flip_idx] = -tgt_mus[:, flip_idx]

    # Mild covariance inflation + jitter
    jitter = (1e-3 * np.eye(d, dtype=np.float32))[None, :, :]
    tgt_Sigmas = cov_scale * src_Sigmas + jitter

    return tgt_mus.astype(np.float32), tgt_Sigmas.astype(np.float32), feat_idx, flip_idx


# ---------------------------------------------------------------------
# 3) Optional: construct a sparse A-mask on precision (banded structure)
# ---------------------------------------------------------------------
def make_band_mask(d, bandwidth=2):
    """
    Binary mask A in {0,1}^{dxd} allowing only a band of width (2*bandwidth+1).
    A[i,j] = 1 if |i-j| <= bandwidth else 0. Diagonal is 1 by construction.
    """
    A = np.zeros((d, d), dtype=np.float32)
    for i in range(d):
        for j in range(max(0, i - bandwidth), min(d, i + bandwidth + 1)):
            A[i, j] = 1.0
    return A


# ---------------------------------------------------------------------
# 4) Main demo: learn priors on Source, transfer to Target, compare ARI
# ---------------------------------------------------------------------
def main():
    key = PRNGKey(0)
    K = 4
    d = 300
    N_src = 2000
    N_tgt = 200

    # Extreme priors: source dominated by cluster 0; target dominated by cluster 3
    pis_src = np.array([0.85, 0.10, 0.04, 0.01], dtype=np.float32)
    pis_tgt = np.array([0.01, 0.04, 0.10, 0.85], dtype=np.float32)

    # --- SOURCE
    Xs, ys, mus_s, Sigmas_s, pis_src_used = simulate_gmm(
        N_src, K, d, seed=42, pis=pis_src
    )

    print("\n=== Fit on SOURCE (to learn priors) ===")
    model_src = Praxis_BGM(
        rng_key=key,
        K=K,
        prior_mus=None,
        prior_Sigmas=None,
        beta=1e-3,
        tol=1e-4,
        max_iters=200,
        verbose=True,
        prior_mus_variance=10.0,
        num_samples=64,
        enforce_mask=False,
        mask_space="precision",
        spd_eps=1e-6,
    )
    model_src.fit(Xs, num_iters=60, batch_size=256, early_stop=True, patience=2)
    mus_post_s, covs_post_s, pis_post_s, _ = model_src.get_posteriors(Xs)

    # --- TARGET params from SOURCE with stronger domain shift & polarity flips
    mus_t, Sigmas_t, feat_idx, flip_idx = make_target_params_from_source(
        np.asarray(mus_post_s),
        np.asarray(covs_post_s),
        frac_feat_shift=0.5,
        global_shift_scale=0.4,
        feat_shift_scale=0.8,
        cov_scale=1.15,
        flip_frac=0.10,
        flip_per_cluster=True,
        seed=123,
    )

    # --- TARGET data with extreme, *reversed* mixture weights
    rng = np.random.default_rng(999)
    pis_t = pis_tgt / pis_tgt.sum()
    zt = rng.choice(K, size=N_tgt, p=pis_t)
    Xt = np.zeros((N_tgt, d), dtype=np.float32)
    for k in range(K):
        nk = np.sum(zt == k)
        if nk > 0:
            Xt[zt == k] = rng.multivariate_normal(mus_t[k], Sigmas_t[k], size=nk)

    # --- Zero-shot baseline (use source model directly on target)
    print("\n=== Direct prediction on TARGET using SOURCE model (no adaptation) ===")
    yhat_direct, _ = model_src.predict(Xt)
    ari_direct = adjusted_rand_score(zt, yhat_direct)
    print(f"[TARGET] ARI direct (source→target): {ari_direct:.3f}")

    # --- NGVI on TARGET with transferred priors (optionally loosen covs)
    print("\n=== Fit on TARGET with TRANSFERRED PRIORS ===")
    cov_loosen = 1.25
    key_src, key_tgt_prior, key_tgt_noprior = split(key, 3)

    model_t_prior = Praxis_BGM(
        rng_key=key_tgt_prior,
        K=K,
        prior_mus=np.array(mus_post_s),
        prior_Sigmas=np.array(covs_post_s) * cov_loosen,
        prior_pis=None,  # let the algorithm learn π
        beta=1e-4,
        tol=1e-4,
        max_iters=300,
        verbose=True,
        prior_mus_variance=1.5,  # slightly looser mean prior
        num_samples=64,
        enforce_mask=False,  # keep mask off unless you know structure
        mask_space="precision",
        spd_eps=1e-6,
    )
    model_t_prior.fit(Xt, num_iters=120, batch_size=128, early_stop=True, patience=3)
    yhat_prior, _ = model_t_prior.predict(Xt)
    ari_prior = adjusted_rand_score(zt, yhat_prior)
    print(f"[TARGET] ARI with transferred priors: {ari_prior:.3f}")

    # --- NGVI on TARGET without priors (baseline)
    print("\n=== Fit on TARGET without priors (baseline) ===")
    model_t_noprior = Praxis_BGM(
        rng_key=key_tgt_noprior,
        K=K,
        prior_mus=None,
        prior_Sigmas=None,
        beta=1e-3,
        tol=1e-4,
        max_iters=300,
        verbose=True,
        prior_mus_variance=10.0,
        num_samples=64,
        enforce_mask=False,
        mask_space="precision",
        spd_eps=1e-6,
    )
    model_t_noprior.fit(Xt, num_iters=120, batch_size=128, early_stop=True, patience=3)
    yhat_noprior, _ = model_t_noprior.predict(Xt)
    ari_noprior = adjusted_rand_score(zt, yhat_noprior)
    print(f"[TARGET] ARI without priors: {ari_noprior:.3f}")

    # --- Optional BF feature scoring
    # BF_mat, feat_scores, top_feats, buckets = model_t_prior.BF_selection(Xt, top_n=15, visual=False)

    # --- Summary
    print("\n=== SUMMARY ===")
    print(f"Source π: {np.round(pis_src_used, 3)}")
    print(f"Target π: {np.round(pis_t, 3)}")
    print(f"Shifted features: {len(feat_idx)} | Flipped features: {len(flip_idx)} (of d={d})")
    print(f"ARI direct (src→tgt): {ari_direct:.3f}")
    print(f"ARI NGVI (with priors): {ari_prior:.3f}")
    print(f"ARI NGVI (no priors):   {ari_noprior:.3f}")


if __name__ == "__main__":
    main()