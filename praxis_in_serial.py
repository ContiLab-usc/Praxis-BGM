#!/usr/bin/env python3
"""Sequential multi-layer Praxis-BGM example.

This script simulates three related omics-style layers with partially aligned
cluster structure. It fits Praxis-BGM on the first layer, then transfers
cluster-wise Gaussian priors forward to layers two and three.
"""

from pathlib import Path
import sys

import numpy as np
from jax.random import PRNGKey, split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_bgm import Praxis_BGM


def sample_categorical(probs, rng):
    """Sample a cluster index from a categorical distribution."""
    return rng.choice(len(probs), p=probs)


def simulate_layer_params(
    K,
    d,
    rng,
    causal_frac=0.2,
    mean_scale=1.0,
    cov_scale=1.0,
    noise_cov_scale=0.4,
    jitter=1e-3,
):
    """Generate per-cluster Gaussian parameters with sparse signal features."""
    n_causal = max(1, int(np.floor(causal_frac * d)))
    causal_idx = rng.choice(d, size=n_causal, replace=False)
    noise_idx = np.array([idx for idx in range(d) if idx not in causal_idx])

    mus = np.zeros((K, d), dtype=np.float32)
    Sigmas = np.zeros((K, d, d), dtype=np.float32)
    mus[:, causal_idx] = rng.normal(0.0, mean_scale, size=(K, n_causal)).astype(np.float32)

    for k in range(K):
        Ac = rng.normal(size=(n_causal, n_causal))
        cov_causal = cov_scale * (Ac @ Ac.T) / max(n_causal, 1)
        cov_causal += jitter * np.eye(n_causal, dtype=np.float32)

        An = rng.normal(size=(len(noise_idx), len(noise_idx)))
        cov_noise = noise_cov_scale * (An @ An.T) / max(len(noise_idx), 1)
        cov_noise += jitter * np.eye(len(noise_idx), dtype=np.float32)

        cov = np.zeros((d, d), dtype=np.float32)
        cov[np.ix_(causal_idx, causal_idx)] = cov_causal.astype(np.float32)
        cov[np.ix_(noise_idx, noise_idx)] = cov_noise.astype(np.float32)
        Sigmas[k] = cov

    return mus, Sigmas


def simulate_layer_data(labels, mus, Sigmas, rng):
    """Sample observations from the cluster-specific Gaussian parameters."""
    labels = np.asarray(labels, dtype=int)
    N = labels.shape[0]
    _, d = mus.shape
    X = np.zeros((N, d), dtype=np.float32)
    for k in range(mus.shape[0]):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            continue
        X[idx] = rng.multivariate_normal(mus[k], Sigmas[k], size=len(idx)).astype(np.float32)
    return X


def simulate_three_layer_clusters(N=600, K=3, seed=0):
    """Simulate correlated cluster labels across three ordered layers."""
    rng = np.random.default_rng(seed)
    base_pi = np.array([0.4, 0.35, 0.25], dtype=np.float32)
    if K != len(base_pi):
        base_pi = np.ones(K, dtype=np.float32) / K

    T12 = np.full((K, K), 0.1 / max(K - 1, 1), dtype=np.float32)
    T23 = np.full((K, K), 0.15 / max(K - 1, 1), dtype=np.float32)
    for k in range(K):
        T12[k, k] = 0.9
        T23[k, k] = 0.85

    Z1 = np.zeros(N, dtype=int)
    Z2 = np.zeros(N, dtype=int)
    Z3 = np.zeros(N, dtype=int)
    for n in range(N):
        Z1[n] = sample_categorical(base_pi, rng)
        Z2[n] = sample_categorical(T12[Z1[n]], rng)
        Z3[n] = sample_categorical(T23[Z2[n]], rng)

    return Z1, Z2, Z3, base_pi, T12, T23


def compute_clusterwise_moments(X, cluster_labels, K, jitter=1e-3):
    """Estimate cluster-wise Gaussian priors from fitted cluster assignments."""
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(cluster_labels, dtype=int)
    N, d = X.shape

    global_mean = X.mean(axis=0)
    X_centered = X - global_mean
    global_cov = (X_centered.T @ X_centered) / max(N - 1, 1)
    global_cov += jitter * np.eye(d, dtype=np.float32)

    mus_prior = np.zeros((K, d), dtype=np.float32)
    Sigmas_prior = np.zeros((K, d, d), dtype=np.float32)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 2:
            mus_prior[k] = global_mean
            Sigmas_prior[k] = global_cov
            continue
        Xk = X[idx]
        mk = Xk.mean(axis=0)
        Xk_centered = Xk - mk
        covk = (Xk_centered.T @ Xk_centered) / max(len(idx) - 1, 1)
        covk += jitter * np.eye(d, dtype=np.float32)
        mus_prior[k] = mk.astype(np.float32)
        Sigmas_prior[k] = covk.astype(np.float32)

    return mus_prior, Sigmas_prior


def fit_bgm_baseline(X, K, seed):
    """Fit a Gaussian-mixture baseline for comparison."""
    gm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        random_state=seed,
        init_params="kmeans",
        max_iter=500,
        n_init=3,
    )
    gm.fit(X)
    return gm.predict(X)


def fit_praxis_layer(key, X, K, prior_mus=None, prior_Sigmas=None, beta=1e-3):
    """Shared helper to fit one Praxis-BGM layer."""
    model = Praxis_BGM(
        rng_key=key,
        K=K,
        prior_mus=prior_mus,
        prior_Sigmas=prior_Sigmas,
        prior_weights=None,
        beta=beta,
        tol=1e-4,
        max_iters=120,
        verbose=True,
        prior_mus_variance=3.0,
        num_samples=16,
        data_precision_int=1,
        likelihood_temp=1.0,
        rho_prec=0.08,
        rho_mu=1.0,
        elbo_eval_freq=1,
    )
    model.fit(X, num_iters=24, batch_size=min(96, X.shape[0]), early_stop=True, patience=3)
    labels, weights = model.predict(X)
    return model, labels, weights


def summarize_clustering(y_true, y_pred):
    """Return a compact clustering-quality summary."""
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
    }


def main():
    N = 450
    K = 3
    d1 = 60
    d2 = 60
    d3 = 60
    sim_seed = 123

    rng = np.random.default_rng(sim_seed)
    key1, key2, key3 = split(PRNGKey(0), 3)

    Z1_true, Z2_true, Z3_true, pi, T12, T23 = simulate_three_layer_clusters(N=N, K=K, seed=sim_seed)
    print("=== True layer relationship ===")
    print("Layer-1 mixture:", np.round(pi, 3))
    print("T12:\n", np.round(T12, 3))
    print("T23:\n", np.round(T23, 3))

    mus1, Sigmas1 = simulate_layer_params(K, d1, rng, mean_scale=1.1, cov_scale=1.0)
    mus2, Sigmas2 = simulate_layer_params(K, d2, rng, mean_scale=0.8, cov_scale=1.2)
    mus3, Sigmas3 = simulate_layer_params(K, d3, rng, mean_scale=0.7, cov_scale=1.4)

    X1 = simulate_layer_data(Z1_true, mus1, Sigmas1, rng)
    X2 = simulate_layer_data(Z2_true, mus2, Sigmas2, rng)
    X3 = simulate_layer_data(Z3_true, mus3, Sigmas3, rng)

    print("\n=== Layer 1: fit without priors ===")
    model1, Z1_hat, _ = fit_praxis_layer(key1, X1, K, beta=1e-3)
    Z1_bgm = fit_bgm_baseline(X1, K, seed=sim_seed)

    mus2_prior, Sigmas2_prior = compute_clusterwise_moments(X2, Z1_hat, K)
    print("\n=== Layer 2: fit with transferred priors from layer 1 ===")
    model2, Z2_hat, _ = fit_praxis_layer(
        key2,
        X2,
        K,
        prior_mus=mus2_prior,
        prior_Sigmas=Sigmas2_prior,
        beta=8e-4,
    )
    key2_no_prior = split(key2, 2)[1]
    model2_noprior, Z2_hat_noprior, _ = fit_praxis_layer(
        key2_no_prior,
        X2,
        K,
        beta=1e-3,
    )
    Z2_bgm = fit_bgm_baseline(X2, K, seed=sim_seed + 1)

    mus3_prior, Sigmas3_prior = compute_clusterwise_moments(X3, Z2_hat, K)
    print("\n=== Layer 3: fit with transferred priors from layer 2 ===")
    model3, Z3_hat, _ = fit_praxis_layer(
        key3,
        X3,
        K,
        prior_mus=mus3_prior,
        prior_Sigmas=Sigmas3_prior,
        beta=8e-4,
    )
    key3_no_prior = split(key3, 2)[1]
    model3_noprior, Z3_hat_noprior, _ = fit_praxis_layer(
        key3_no_prior,
        X3,
        K,
        beta=1e-3,
    )
    Z3_bgm = fit_bgm_baseline(X3, K, seed=sim_seed + 2)

    layer1_praxis = summarize_clustering(Z1_true, Z1_hat)
    layer2_prior = summarize_clustering(Z2_true, Z2_hat)
    layer2_noprior = summarize_clustering(Z2_true, Z2_hat_noprior)
    layer3_prior = summarize_clustering(Z3_true, Z3_hat)
    layer3_noprior = summarize_clustering(Z3_true, Z3_hat_noprior)

    print("\n=== Layer-wise clustering quality ===")
    print(
        f"Layer 1 | Praxis ARI={layer1_praxis['ari']:.3f}, "
        f"NMI={layer1_praxis['nmi']:.3f} | "
        f"BGM ARI={adjusted_rand_score(Z1_true, Z1_bgm):.3f}"
    )
    print(
        f"Layer 2 | Praxis with priors ARI={layer2_prior['ari']:.3f}, "
        f"NMI={layer2_prior['nmi']:.3f} | "
        f"Praxis no priors ARI={layer2_noprior['ari']:.3f}, "
        f"NMI={layer2_noprior['nmi']:.3f} | "
        f"BGM ARI={adjusted_rand_score(Z2_true, Z2_bgm):.3f}"
    )
    print(
        f"Layer 3 | Praxis with priors ARI={layer3_prior['ari']:.3f}, "
        f"NMI={layer3_prior['nmi']:.3f} | "
        f"Praxis no priors ARI={layer3_noprior['ari']:.3f}, "
        f"NMI={layer3_noprior['nmi']:.3f} | "
        f"BGM ARI={adjusted_rand_score(Z3_true, Z3_bgm):.3f}"
    )

    print("\n=== Cross-layer agreement ===")
    print(f"True labels   | ARI(Z1, Z2) = {adjusted_rand_score(Z1_true, Z2_true):.3f}")
    print(f"True labels   | ARI(Z2, Z3) = {adjusted_rand_score(Z2_true, Z3_true):.3f}")
    print(f"Praxis labels | ARI(Z1, Z2) = {adjusted_rand_score(Z1_hat, Z2_hat):.3f}")
    print(f"Praxis labels | ARI(Z2, Z3) = {adjusted_rand_score(Z2_hat, Z3_hat):.3f}")

    print("\n=== Final ELBO snapshots ===")
    print(f"Layer 1 last MC ELBO: {model1.elbo_history[-1]:.3f}" if model1.elbo_history else "Layer 1 ELBO unavailable")
    print(f"Layer 2 last MC ELBO: {model2.elbo_history[-1]:.3f}" if model2.elbo_history else "Layer 2 ELBO unavailable")
    print(f"Layer 3 last MC ELBO: {model3.elbo_history[-1]:.3f}" if model3.elbo_history else "Layer 3 ELBO unavailable")


if __name__ == "__main__":
    main()
