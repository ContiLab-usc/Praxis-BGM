#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sequential_multiomics_praxis_vs_bgm.py

Simulate 3 omics layers for the same subjects with correlated but non-identical
cluster labels (Z1 -> Z2 -> Z3 via transition matrices).

Then:
  1) Run Praxis-BGM on layer 1 with no prior.
  2) Given layer-1 Praxis clusters, build cluster-specific priors for layer 2
     and run Praxis-BGM.
  3) Given layer-2 Praxis clusters, build cluster-specific priors for layer 3
     and run Praxis-BGM.
In addition, for EACH layer:
  - Run a separate baseline BGM (here: sklearn GaussianMixture).
  - Compare Praxis vs BGM vs ground truth for that layer.

Finally, analyze cross-layer correlations between the three clusterings.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Import Praxis_BGM (adjust this import path to match your package structure)
from Praxis_BGM import Praxis_BGM


# ---------------------------------------------------------------------
# 1) Utilities: categorical sampling and Gaussian mixtures
# ---------------------------------------------------------------------
def sample_categorical(probs, rng):
    """
    Sample from a categorical distribution with probability vector `probs`.
    probs: (K,) array
    rng: np.random.Generator
    """
    return rng.choice(len(probs), p=probs)


def simulate_layer_params(
    K,
    d,
    rng,
    causal_frac=0.20,
    mean_scale=2.0,
    cov_scale=1.0,
    noise_cov_scale=0.5,
    jitter=1e-3
):
    """
    Simulate GMM parameters (mus, Sigmas) for one omics layer.
    
    Only `causal_frac` proportion of features contribute to cluster separation.
    The remaining features are pure noise (same distribution across clusters).
    
    Args
    ----
    K : int
        Number of clusters.
    d : int
        Dimensionality.
    rng : np.random.Generator
    
    causal_frac : float
        Fraction of features that differ across clusters (default: 10%).
        
    mean_scale : float
        Scale for cluster-specific mean differences on causal features.
        
    cov_scale : float
        Scale for covariance variability for causal features.
        
    noise_cov_scale : float
        Scale for covariance for noise features (smaller).
        
    Returns
    -------
    mus : (K, d)
    Sigmas : (K, d, d)
    causal_idx : indices of causal features
    """
    
    # ---------------------------
    # 1) Select causal features
    # ---------------------------
    m = max(1, int(np.floor(causal_frac * d)))
    causal_idx = rng.choice(d, size=m, replace=False)
    noise_idx = np.array([i for i in range(d) if i not in causal_idx])

    # ---------------------------
    # 2) Initialize parameters
    # ---------------------------
    mus = np.zeros((K, d), dtype=np.float32)
    Sigmas = np.zeros((K, d, d), dtype=np.float32)

    # ---------------------------
    # 3) Cluster-specific means (only for causal features)
    # ---------------------------
    # mus[k, causal] ~ Normal(0, mean_scale^2)
    mus[:, causal_idx] = rng.normal(
        loc=0.0, scale=mean_scale, size=(K, m)
    ).astype(np.float32)

    # ---------------------------
    # 4) Covariances
    # ---------------------------
    for k in range(K):
        # ---- Causal block: cluster-specific covariance ----
        Ac = rng.normal(size=(m, m))
        cov_causal = cov_scale * (Ac @ Ac.T) / m
        cov_causal += jitter * np.eye(m)

        # ---- Noise block: identical for all clusters ----
        An = rng.normal(size=(len(noise_idx), len(noise_idx)))
        cov_noise = noise_cov_scale * (An @ An.T) / len(noise_idx)
        cov_noise += jitter * np.eye(len(noise_idx))

        # ---- Combine into full covariance matrix ----
        cov = np.zeros((d, d), dtype=np.float32)
        # fill causal block
        cov[np.ix_(causal_idx, causal_idx)] = cov_causal
        # fill noise block
        cov[np.ix_(noise_idx, noise_idx)] = cov_noise

        Sigmas[k] = cov.astype(np.float32)

    return mus, Sigmas


def simulate_layer_data(Z, mus, Sigmas, rng):
    """
    Given cluster labels Z (N,), and per-cluster mus, Sigmas,
    simulate X ~ N(mus[Z], Sigmas[Z]).

    Returns X: (N, d)
    """
    N = len(Z)
    K, d = mus.shape
    X = np.zeros((N, d), dtype=np.float32)
    for k in range(K):
        idx = np.where(Z == k)[0]
        if len(idx) == 0:
            continue
        X[idx] = rng.multivariate_normal(mus[k], Sigmas[k], size=len(idx))
    return X.astype(np.float32)


# ---------------------------------------------------------------------
# 2) Simulate 3-layer cluster structure with biological ordering
# ---------------------------------------------------------------------
def simulate_three_layer_clusters(N=600, K=3, seed=0):
    """
    Simulate Z1, Z2, Z3 for N subjects with K clusters each.
    Z1 ~ Categorical(pi)
    Z2 | Z1 ~ T12 (transition matrix close to identity)
    Z3 | Z2 ~ T23 (transition matrix close to identity)

    Returns:
        Z1, Z2, Z3: int arrays (N,)
        T12, T23: transition matrices (K,K)
        pi: base mixture (K,)
    """
    rng = np.random.default_rng(seed)

    # Base mixture for layer 1
    pi = np.array([0.4, 0.35, 0.25], dtype=np.float32)
    if len(pi) != K:
        pi = np.ones(K, dtype=np.float32) / K

    # Transitions: strong diagonal to enforce biological ordering / influence
    T12 = np.full((K, K), 0.1 / (K - 1), dtype=np.float32)
    T23 = np.full((K, K), 0.15 / (K - 1), dtype=np.float32)
    for k in range(K):
        T12[k, k] = 0.9
        T23[k, k] = 0.85

    Z1 = np.zeros(N, dtype=int)
    Z2 = np.zeros(N, dtype=int)
    Z3 = np.zeros(N, dtype=int)

    for n in range(N):
        Z1[n] = sample_categorical(pi, rng)
        Z2[n] = sample_categorical(T12[Z1[n]], rng)
        Z3[n] = sample_categorical(T23[Z2[n]], rng)

    return Z1, Z2, Z3, pi, T12, T23


# ---------------------------------------------------------------------
# 3) Build empirical cluster-specific priors from another layer's clusters
# ---------------------------------------------------------------------
def compute_clusterwise_moments(X, cluster_labels, K, jitter=1e-3):
    """
    Given data X (N,d) and cluster_labels (N,), compute empirical
    means and covariances for each cluster k=0..K-1.

    If a cluster has fewer than 2 points, fall back to global moments.
    Returns:
        mus_prior: (K,d)
        Sigmas_prior: (K,d,d)
    """
    X = np.asarray(X, dtype=np.float32)
    N, d = X.shape
    cluster_labels = np.asarray(cluster_labels, dtype=int)

    global_mean = X.mean(axis=0, keepdims=False)
    X_centered = X - global_mean
    global_cov = (X_centered.T @ X_centered) / max(N - 1, 1)
    global_cov = global_cov + jitter * np.eye(d, dtype=np.float32)

    mus_prior = np.zeros((K, d), dtype=np.float32)
    Sigmas_prior = np.zeros((K, d, d), dtype=np.float32)

    for k in range(K):
        idx = np.where(cluster_labels == k)[0]
        if len(idx) < 2:
            # fallback to global
            mus_prior[k] = global_mean
            Sigmas_prior[k] = global_cov
        else:
            Xk = X[idx]
            mk = Xk.mean(axis=0, keepdims=False)
            Xk_centered = Xk - mk
            covk = (Xk_centered.T @ Xk_centered) / max(len(idx) - 1, 1)
            covk = covk + jitter * np.eye(d, dtype=np.float32)
            mus_prior[k] = mk.astype(np.float32)
            Sigmas_prior[k] = covk.astype(np.float32)

    return mus_prior, Sigmas_prior


# ---------------------------------------------------------------------
# 4) Helper: fit baseline BGM (here: sklearn GaussianMixture)
# ---------------------------------------------------------------------
def fit_bgm_baseline(X, K, seed=0):
    """
    Fit a baseline Bayesian Gaussian mixture model to X.
    Here we use sklearn.mixture.GaussianMixture as a stand-in.
    If you have your own BGM implementation, plug it in here.

    Returns:
        z_bgm: (N,) inferred labels
    """
    gm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        random_state=seed,
        init_params="kmeans",
        max_iter=500,
        n_init=3,
    )
    gm.fit(X)
    z_bgm = gm.predict(X)
    return z_bgm


# ---------------------------------------------------------------------
# 5) Main demo: simulate 3 layers, fit Praxis sequentially + BGM baselines
# ---------------------------------------------------------------------
def main():
    # -----------------------------
    # Config
    # -----------------------------
    N = 600
    K = 3
    d1 = 100   # dimensionality of layer 1 (e.g., transcriptomics)
    d2 = 100   # dimensionality of layer 2 (e.g., proteomics)
    d3 = 100 # dimensionality of layer 3 (e.g., metabolomics)

    sim_seed = 123
    rng = np.random.default_rng(sim_seed)

    # JAX RNG keys for Praxis-BGM
    key = PRNGKey(0)
    key1, key2, key3 = split(key, 3)

    # -----------------------------
    # Step 1: simulate latent clusters Z1, Z2, Z3
    # -----------------------------
    Z1_true, Z2_true, Z3_true, pi, T12, T23 = simulate_three_layer_clusters(
        N=N, K=K, seed=sim_seed
    )

    print("=== True mixture and transitions ===")
    print("pi (layer 1):", np.round(pi, 3))
    print("T12 (Z1 -> Z2):\n", np.round(T12, 3))
    print("T23 (Z2 -> Z3):\n", np.round(T23, 3))

    # -----------------------------
    # Step 2: simulate multi-omics data for each layer
    # -----------------------------
    mus1, Sigmas1 = simulate_layer_params(K, d1, rng, mean_scale=1, cov_scale=1.0)
    mus2, Sigmas2 = simulate_layer_params(K, d2, rng, mean_scale=0.5, cov_scale=1.2)
    mus3, Sigmas3 = simulate_layer_params(K, d3, rng, mean_scale=0.5, cov_scale=1.4)

    X1 = simulate_layer_data(Z1_true, mus1, Sigmas1, rng)  # (N,d1)
    X2 = simulate_layer_data(Z2_true, mus2, Sigmas2, rng)  # (N,d2)
    X3 = simulate_layer_data(Z3_true, mus3, Sigmas3, rng)  # (N,d3)

    # -----------------------------
    # Step 3: Praxis-BGM on Layer 1 (no priors) + BGM baseline
    # -----------------------------
    print("\n=== Layer 1: Praxis-BGM (no priors) and BGM baseline ===")
    model1 = Praxis_BGM(
        rng_key=key1,
        K=K,
        prior_mus=None,
        prior_Sigmas=None,
        prior_pis=None,
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
    model1.fit(X1, num_iters=150, batch_size=128, early_stop=True, patience=5)
    Z1_hat, _ = model1.predict(X1)

    # BGM baseline on layer 1
    Z1_bgm = fit_bgm_baseline(X1, K=K, seed=sim_seed)

    ari1_praxis = adjusted_rand_score(Z1_true, Z1_hat)
    nmi1_praxis = normalized_mutual_info_score(Z1_true, Z1_hat)
    ari1_bgm = adjusted_rand_score(Z1_true, Z1_bgm)
    nmi1_bgm = normalized_mutual_info_score(Z1_true, Z1_bgm)
    ari1_praxis_bgm = adjusted_rand_score(Z1_hat, Z1_bgm)

    print(f"[Layer 1] Praxis vs truth: ARI = {ari1_praxis:.3f}, NMI = {nmi1_praxis:.3f}")
    print(f"[Layer 1] BGM   vs truth: ARI = {ari1_bgm:.3f}, NMI = {nmi1_bgm:.3f}")
    print(f"[Layer 1] Praxis vs BGM:  ARI = {ari1_praxis_bgm:.3f}")

    # -----------------------------
    # Step 4: Priors for Layer 2 from Layer 1 Praxis clusters + BGM on layer 2
    # -----------------------------
    print("\n=== Building priors for Layer 2 from Layer-1 Praxis clusters ===")
    mus2_prior, Sigmas2_prior = compute_clusterwise_moments(X2, Z1_hat, K, jitter=1e-3)

    print("\n=== Layer 2: Praxis-BGM (with priors) and BGM baseline ===")
    model2 = Praxis_BGM(
        rng_key=key2,
        K=K,
        prior_mus=mus2_prior,
        prior_Sigmas=Sigmas2_prior,
        prior_pis=None,
        beta=1e-4,
        tol=1e-4,
        max_iters=300,
        verbose=True,
        prior_mus_variance=10.0,
        num_samples=64,
        enforce_mask=False,
        mask_space="precision",
        spd_eps=1e-6,
    )
    model2.fit(X2, num_iters=200, batch_size=128, early_stop=True, patience=5)
    Z2_hat, _ = model2.predict(X2)

    # BGM baseline on layer 2
    Z2_bgm = fit_bgm_baseline(X2, K=K, seed=sim_seed + 1)

    ari2_praxis = adjusted_rand_score(Z2_true, Z2_hat)
    nmi2_praxis = normalized_mutual_info_score(Z2_true, Z2_hat)
    ari2_bgm = adjusted_rand_score(Z2_true, Z2_bgm)
    nmi2_bgm = normalized_mutual_info_score(Z2_true, Z2_bgm)
    ari2_praxis_bgm = adjusted_rand_score(Z2_hat, Z2_bgm)

    print(f"[Layer 2] Praxis vs truth: ARI = {ari2_praxis:.3f}, NMI = {nmi2_praxis:.3f}")
    print(f"[Layer 2] BGM   vs truth: ARI = {ari2_bgm:.3f}, NMI = {nmi2_bgm:.3f}")
    print(f"[Layer 2] Praxis vs BGM:  ARI = {ari2_praxis_bgm:.3f}")

    # -----------------------------
    # Step 5: Priors for Layer 3 from Layer 2 Praxis clusters + BGM on layer 3
    # -----------------------------
    print("\n=== Building priors for Layer 3 from Layer-2 Praxis clusters ===")
    mus3_prior, Sigmas3_prior = compute_clusterwise_moments(X3, Z2_hat, K, jitter=1e-3)

    print("\n=== Layer 3: Praxis-BGM (with priors) and BGM baseline ===")
    model3 = Praxis_BGM(
        rng_key=key3,
        K=K,
        prior_mus=mus3_prior,
        prior_Sigmas=Sigmas3_prior,
        prior_pis=None,
        beta=1e-4,
        tol=1e-4,
        max_iters=300,
        verbose=True,
        prior_mus_variance=10.0,
        num_samples=64,
        enforce_mask=False,
        mask_space="precision",
        spd_eps=1e-6,
    )
    model3.fit(X3, num_iters=200, batch_size=128, early_stop=True, patience=5)
    Z3_hat, _ = model3.predict(X3)

    # BGM baseline on layer 3
    Z3_bgm = fit_bgm_baseline(X3, K=K, seed=sim_seed + 2)

    ari3_praxis = adjusted_rand_score(Z3_true, Z3_hat)
    nmi3_praxis = normalized_mutual_info_score(Z3_true, Z3_hat)
    ari3_bgm = adjusted_rand_score(Z3_true, Z3_bgm)
    nmi3_bgm = normalized_mutual_info_score(Z3_true, Z3_bgm)
    ari3_praxis_bgm = adjusted_rand_score(Z3_hat, Z3_bgm)

    print(f"[Layer 3] Praxis vs truth: ARI = {ari3_praxis:.3f}, NMI = {nmi3_praxis:.3f}")
    print(f"[Layer 3] BGM   vs truth: ARI = {ari3_bgm:.3f}, NMI = {nmi3_bgm:.3f}")
    print(f"[Layer 3] Praxis vs BGM:  ARI = {ari3_praxis_bgm:.3f}")

    # -----------------------------
    # Step 6: Cross-layer correlations (true vs Praxis estimates)
    # -----------------------------
    print("\n=== Cross-layer correlations (true clusters) ===")
    print(f"ARI(Z1_true, Z2_true) = {adjusted_rand_score(Z1_true, Z2_true):.3f}")
    print(f"ARI(Z2_true, Z3_true) = {adjusted_rand_score(Z2_true, Z3_true):.3f}")
    print(f"ARI(Z1_true, Z3_true) = {adjusted_rand_score(Z1_true, Z3_true):.3f}")

    print("\n=== Cross-layer correlations (Praxis clusters) ===")
    print(f"ARI(Z1_hat, Z2_hat) = {adjusted_rand_score(Z1_hat, Z2_hat):.3f}")
    print(f"ARI(Z2_hat, Z3_hat) = {adjusted_rand_score(Z2_hat, Z3_hat):.3f}")
    print(f"ARI(Z1_hat, Z3_hat) = {adjusted_rand_score(Z1_hat, Z3_hat):.3f}")

    print("\n=== Cross-layer alignment: true vs Praxis across layers ===")
    print(f"ARI(Z1_true, Z2_hat) = {adjusted_rand_score(Z1_true, Z2_hat):.3f}")
    print(f"ARI(Z2_true, Z1_hat) = {adjusted_rand_score(Z2_true, Z1_hat):.3f}")
    print(f"ARI(Z2_true, Z3_hat) = {adjusted_rand_score(Z2_true, Z3_hat):.3f}")
    print(f"ARI(Z3_true, Z2_hat) = {adjusted_rand_score(Z3_true, Z2_hat):.3f}")


if __name__ == "__main__":
    main()