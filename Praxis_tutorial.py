#!/usr/bin/env python3
"""Tutorial simulation for the packaged Praxis library.

This script demonstrates a simple transfer-learning workflow:

1. Fit Praxis on a larger source dataset.
2. Build a shifted target domain from the learned source posteriors.
3. Compare direct source prediction, target fitting with transferred priors,
   and target fitting without priors.

The tutorial supports both a moderate-dimensional example and a high-dimensional
example selected via ``--profile``.
"""

import argparse
from pathlib import Path
import sys

import jax
import numpy as np
from jax.random import PRNGKey, split
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from praxis_bgm import Praxis_BGM


def make_block_cov(d, block=6, on_diag_var=1.0, off_diag=0.25, jitter=1e-3, seed=0):
    """Construct a block-correlated covariance matrix for simulation."""
    rng = np.random.default_rng(seed)
    Sigma = np.eye(d, dtype=np.float32) * on_diag_var
    for start in range(0, d, block):
        end = min(start + block, d)
        if end - start >= 2:
            block_idx = slice(start, end)
            block_noise = off_diag * (
                np.ones((end - start, end - start), dtype=np.float32)
                - np.eye(end - start, dtype=np.float32)
            )
            Sigma[block_idx, block_idx] += block_noise
    Sigma += jitter * np.eye(d, dtype=np.float32)

    # Add a tiny random perturbation so clusters are not too perfectly structured.
    Sigma += 1e-4 * rng.normal(size=(d, d)).astype(np.float32)
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma += 1e-3 * np.eye(d, dtype=np.float32)
    return Sigma.astype(np.float32)


def simulate_gmm(N, K, d, seed=0, pis=None):
    """Simulate a Gaussian mixture dataset with mildly heterogeneous covariance."""
    rng = np.random.default_rng(seed)
    base_dirs = rng.normal(size=(K, d)).astype(np.float32)
    base_dirs /= np.linalg.norm(base_dirs, axis=1, keepdims=True) + 1e-12
    mus = 1.5 * base_dirs
    Sigmas = np.stack(
        [
            make_block_cov(
                d,
                block=6,
                on_diag_var=1.0 + 0.2 * k,
                off_diag=0.22,
                seed=seed + 10 * k,
            )
            for k in range(K)
        ],
        axis=0,
    )

    if pis is None:
        pis = rng.dirichlet(alpha=np.ones(K)).astype(np.float32)
    else:
        pis = np.asarray(pis, dtype=np.float32)
        pis = pis / pis.sum()

    z = rng.choice(K, size=N, p=pis)
    X = np.zeros((N, d), dtype=np.float32)
    for k in range(K):
        nk = np.sum(z == k)
        if nk > 0:
            X[z == k] = rng.multivariate_normal(mus[k], Sigmas[k], size=nk).astype(np.float32)

    return X, z.astype(int), mus.astype(np.float32), Sigmas.astype(np.float32), pis.astype(np.float32)


def make_target_params_from_source(
    src_mus,
    src_Sigmas,
    frac_feat_shift=0.25,
    global_shift_scale=0.35,
    feat_shift_scale=0.55,
    cov_scale=1.10,
    flip_frac=0.10,
    flip_per_cluster=True,
    seed=123,
):
    """Create a shifted target domain from source posterior parameters."""
    src_mus = np.asarray(src_mus, dtype=np.float32)
    src_Sigmas = np.asarray(src_Sigmas, dtype=np.float32)
    rng = np.random.default_rng(seed)
    K, d = src_mus.shape

    global_shift = rng.normal(scale=global_shift_scale, size=(d,)).astype(np.float32)

    n_shift = max(1, int(np.floor(frac_feat_shift * d)))
    feat_idx = rng.choice(np.arange(d), size=n_shift, replace=False)
    feature_shift = np.zeros((K, d), dtype=np.float32)
    feature_shift[:, feat_idx] = rng.normal(scale=feat_shift_scale, size=(K, n_shift)).astype(np.float32)
    tgt_mus = src_mus + global_shift[None, :] + feature_shift

    n_flip = max(1, int(np.floor(flip_frac * d)))
    flip_idx = rng.choice(np.arange(d), size=n_flip, replace=False)
    if flip_per_cluster:
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(K,), replace=True)
        tgt_mus[:, flip_idx] = tgt_mus[:, flip_idx] * signs[:, None]
    else:
        tgt_mus[:, flip_idx] = -tgt_mus[:, flip_idx]

    jitter = (1e-3 * np.eye(d, dtype=np.float32))[None, :, :]
    tgt_Sigmas = cov_scale * src_Sigmas + jitter
    return tgt_mus.astype(np.float32), tgt_Sigmas.astype(np.float32), feat_idx, flip_idx


def print_section(title):
    """Print a visually clear section header."""
    print(f"\n=== {title} ===")


def print_mapping(title, mapping):
    """Pretty-print a flat dictionary of tutorial settings or metrics."""
    print_section(title)
    for key, value in mapping.items():
        print(f"{key}: {value}")


def get_tutorial_profile(profile_name):
    """Return a predefined tutorial configuration profile."""
    profiles = {
        "standard": {
            "profile_name": "standard",
            "profile_description": "Moderate-dimensional transfer-learning example.",
            "simulation_config": {
                "seed": 0,
                "K": 3,
                "d": 24,
                "N_src": 360,
                "N_tgt": 150,
                "source_mixture_weights": [0.70, 0.20, 0.10],
                "target_mixture_weights": [0.10, 0.25, 0.65],
                "frac_feat_shift": 0.18,
                "global_shift_scale": 0.22,
                "feat_shift_scale": 0.30,
                "cov_scale": 1.06,
                "flip_frac": 0.04,
            },
            "source_hparams": {
                "beta": 1e-3,
                "tol": 1e-4,
                "max_iters": 60,
                "num_iters": 18,
                "batch_size": 96,
                "num_samples": 16,
                "prior_mus_variance": 3.0,
                "likelihood_temp": 1.0,
                "rho_prec": 0.08,
                "rho_mu": 1.0,
                "elbo_eval_freq": 1,
                "data_precision_int": 1,
                "early_stop": True,
                "patience": 2,
            },
            "target_transfer_hparams": {
                "beta": 8e-4,
                "tol": 1e-4,
                "max_iters": 80,
                "num_iters": 24,
                "batch_size": 96,
                "num_samples": 16,
                "prior_mus_variance": 1.0,
                "likelihood_temp": 1.0,
                "rho_prec": 0.08,
                "rho_mu": 1.0,
                "elbo_eval_freq": 1,
                "data_precision_int": 1,
                "early_stop": True,
                "patience": 3,
                "covariance_transfer_scale": 1.08,
            },
            "target_baseline_hparams": {
                "beta": 1e-3,
                "tol": 1e-4,
                "max_iters": 80,
                "num_iters": 24,
                "batch_size": 96,
                "num_samples": 16,
                "prior_mus_variance": 3.0,
                "likelihood_temp": 1.0,
                "rho_prec": 0.08,
                "rho_mu": 1.0,
                "elbo_eval_freq": 1,
                "data_precision_int": 1,
                "early_stop": True,
                "patience": 3,
            },
        },
        "high_dim": {
            "profile_name": "high_dim",
            "profile_description": "High-dimensional, low-target-sample regime where transferred priors help strongly.",
            "simulation_config": {
                "seed": 0,
                "K": 3,
                "d": 160,
                "N_src": 700,
                "N_tgt": 60,
                "source_mixture_weights": [0.70, 0.20, 0.10],
                "target_mixture_weights": [0.10, 0.25, 0.65],
                "frac_feat_shift": 0.10,
                "global_shift_scale": 0.15,
                "feat_shift_scale": 0.18,
                "cov_scale": 1.03,
                "flip_frac": 0.01,
            },
            "source_hparams": {
                "beta": 1e-3,
                "tol": 1e-4,
                "max_iters": 60,
                "num_iters": 14,
                "batch_size": 160,
                "num_samples": 16,
                "prior_mus_variance": 3.0,
                "likelihood_temp": 1.0,
                "rho_prec": 0.08,
                "rho_mu": 1.0,
                "elbo_eval_freq": 1,
                "data_precision_int": 1,
                "early_stop": True,
                "patience": 2,
            },
            "target_transfer_hparams": {
                "beta": 8e-4,
                "tol": 1e-4,
                "max_iters": 80,
                "num_iters": 18,
                "batch_size": 48,
                "num_samples": 16,
                "prior_mus_variance": 0.9,
                "likelihood_temp": 1.0,
                "rho_prec": 0.08,
                "rho_mu": 1.0,
                "elbo_eval_freq": 1,
                "data_precision_int": 1,
                "early_stop": True,
                "patience": 3,
                "covariance_transfer_scale": 1.03,
            },
            "target_baseline_hparams": {
                "beta": 1e-3,
                "tol": 1e-4,
                "max_iters": 80,
                "num_iters": 18,
                "batch_size": 48,
                "num_samples": 16,
                "prior_mus_variance": 3.0,
                "likelihood_temp": 1.0,
                "rho_prec": 0.08,
                "rho_mu": 1.0,
                "elbo_eval_freq": 1,
                "data_precision_int": 1,
                "early_stop": True,
                "patience": 3,
            },
        },
    }

    if profile_name not in profiles:
        raise ValueError(f"Unknown tutorial profile '{profile_name}'. Available profiles: {sorted(profiles)}")
    return profiles[profile_name]


def run_tutorial(profile_name="standard"):
    """Run the end-to-end tutorial simulation and return a results dictionary."""
    key = PRNGKey(0)
    profile = get_tutorial_profile(profile_name)
    simulation_config = profile["simulation_config"]
    source_hparams = profile["source_hparams"]
    target_transfer_hparams = profile["target_transfer_hparams"]
    target_baseline_hparams = profile["target_baseline_hparams"]

    print_mapping(
        "Tutorial Profile",
        {
            "profile_name": profile["profile_name"],
            "profile_description": profile["profile_description"],
        },
    )

    print_mapping("Simulation Hyperparameters", simulation_config)
    print_mapping("Source Model Hyperparameters", source_hparams)
    print_mapping("Target Transfer Hyperparameters", target_transfer_hparams)
    print_mapping("Target Baseline Hyperparameters", target_baseline_hparams)

    K = simulation_config["K"]
    d = simulation_config["d"]
    N_src = simulation_config["N_src"]
    N_tgt = simulation_config["N_tgt"]
    pis_src = np.array(simulation_config["source_mixture_weights"], dtype=np.float32)
    pis_tgt = np.array(simulation_config["target_mixture_weights"], dtype=np.float32)

    Xs, ys, _, _, pis_src_used = simulate_gmm(N_src, K, d, seed=42, pis=pis_src)

    print_section("Fit on SOURCE")
    model_src = Praxis_BGM(
        rng_key=key,
        K=K,
        prior_mus=None,
        prior_Sigmas=None,
        beta=source_hparams["beta"],
        tol=source_hparams["tol"],
        max_iters=source_hparams["max_iters"],
        verbose=True,
        prior_mus_variance=source_hparams["prior_mus_variance"],
        num_samples=source_hparams["num_samples"],
        data_precision_int=source_hparams["data_precision_int"],
        likelihood_temp=source_hparams["likelihood_temp"],
        rho_prec=source_hparams["rho_prec"],
        rho_mu=source_hparams["rho_mu"],
        elbo_eval_freq=source_hparams["elbo_eval_freq"],
    )
    model_src.fit(
        Xs,
        num_iters=source_hparams["num_iters"],
        batch_size=source_hparams["batch_size"],
        early_stop=source_hparams["early_stop"],
        patience=source_hparams["patience"],
    )
    mus_post_s, covs_post_s, _, _ = model_src.get_posteriors(Xs)

    mus_t, Sigmas_t, feat_idx, flip_idx = make_target_params_from_source(
        np.asarray(mus_post_s),
        np.asarray(covs_post_s),
        frac_feat_shift=simulation_config["frac_feat_shift"],
        global_shift_scale=simulation_config["global_shift_scale"],
        feat_shift_scale=simulation_config["feat_shift_scale"],
        cov_scale=simulation_config["cov_scale"],
        flip_frac=simulation_config["flip_frac"],
        flip_per_cluster=True,
        seed=123,
    )

    rng = np.random.default_rng(999)
    zt = rng.choice(K, size=N_tgt, p=pis_tgt / pis_tgt.sum())
    Xt = np.zeros((N_tgt, d), dtype=np.float32)
    for k in range(K):
        nk = np.sum(zt == k)
        if nk > 0:
            Xt[zt == k] = rng.multivariate_normal(mus_t[k], Sigmas_t[k], size=nk).astype(np.float32)

    print_section("Direct SOURCE -> TARGET Prediction")
    yhat_direct, _ = model_src.predict(Xt)
    ari_direct = adjusted_rand_score(zt, yhat_direct)
    print(f"ARI direct (source to target, no adaptation): {ari_direct:.3f}")

    _, key_tgt_prior, key_tgt_noprior = split(key, 3)

    print_section("Fit on TARGET with Transferred Priors")
    model_t_prior = Praxis_BGM(
        rng_key=key_tgt_prior,
        K=K,
        prior_mus=np.array(mus_post_s),
        prior_Sigmas=np.array(covs_post_s) * target_transfer_hparams["covariance_transfer_scale"],
        beta=target_transfer_hparams["beta"],
        tol=target_transfer_hparams["tol"],
        max_iters=target_transfer_hparams["max_iters"],
        verbose=True,
        prior_mus_variance=target_transfer_hparams["prior_mus_variance"],
        num_samples=target_transfer_hparams["num_samples"],
        data_precision_int=target_transfer_hparams["data_precision_int"],
        likelihood_temp=target_transfer_hparams["likelihood_temp"],
        rho_prec=target_transfer_hparams["rho_prec"],
        rho_mu=target_transfer_hparams["rho_mu"],
        elbo_eval_freq=target_transfer_hparams["elbo_eval_freq"],
    )
    model_t_prior.fit(
        Xt,
        num_iters=target_transfer_hparams["num_iters"],
        batch_size=target_transfer_hparams["batch_size"],
        early_stop=target_transfer_hparams["early_stop"],
        patience=target_transfer_hparams["patience"],
    )
    yhat_prior, learned_pis_prior = model_t_prior.predict(Xt)
    ari_prior = adjusted_rand_score(zt, yhat_prior)

    print_section("Fit on TARGET without Priors")
    model_t_noprior = Praxis_BGM(
        rng_key=key_tgt_noprior,
        K=K,
        prior_mus=None,
        prior_Sigmas=None,
        beta=target_baseline_hparams["beta"],
        tol=target_baseline_hparams["tol"],
        max_iters=target_baseline_hparams["max_iters"],
        verbose=True,
        prior_mus_variance=target_baseline_hparams["prior_mus_variance"],
        num_samples=target_baseline_hparams["num_samples"],
        data_precision_int=target_baseline_hparams["data_precision_int"],
        likelihood_temp=target_baseline_hparams["likelihood_temp"],
        rho_prec=target_baseline_hparams["rho_prec"],
        rho_mu=target_baseline_hparams["rho_mu"],
        elbo_eval_freq=target_baseline_hparams["elbo_eval_freq"],
    )
    model_t_noprior.fit(
        Xt,
        num_iters=target_baseline_hparams["num_iters"],
        batch_size=target_baseline_hparams["batch_size"],
        early_stop=target_baseline_hparams["early_stop"],
        patience=target_baseline_hparams["patience"],
    )
    yhat_noprior, learned_pis_noprior = model_t_noprior.predict(Xt)
    ari_noprior = adjusted_rand_score(zt, yhat_noprior)

    result_summary = {
        "profile_name": profile["profile_name"],
        "source_pi_used": np.round(pis_src_used, 3).tolist(),
        "target_pi_true": np.round(pis_tgt, 3).tolist(),
        "target_pi_learned_with_priors": np.round(learned_pis_prior, 3).tolist(),
        "target_pi_learned_without_priors": np.round(learned_pis_noprior, 3).tolist(),
        "shifted_feature_count": int(len(feat_idx)),
        "flipped_feature_count": int(len(flip_idx)),
        "ari_direct_source_to_target": round(float(ari_direct), 3),
        "ari_target_with_transferred_priors": round(float(ari_prior), 3),
        "ari_target_without_priors": round(float(ari_noprior), 3),
        "source_last_mc_elbo": None if not model_src.elbo_history else round(float(model_src.elbo_history[-1]), 3),
        "target_prior_last_mc_elbo": None
        if not model_t_prior.elbo_history
        else round(float(model_t_prior.elbo_history[-1]), 3),
        "target_no_prior_last_mc_elbo": None
        if not model_t_noprior.elbo_history
        else round(float(model_t_noprior.elbo_history[-1]), 3),
    }

    print_mapping("Tutorial Results", result_summary)
    return result_summary


def main():
    """Entry point used when running the tutorial as a script."""
    parser = argparse.ArgumentParser(description="Run the packaged Praxis tutorial simulation.")
    parser.add_argument(
        "--profile",
        default="standard",
        choices=["standard", "high_dim"],
        help="Tutorial configuration profile to run.",
    )
    args = parser.parse_args()
    run_tutorial(profile_name=args.profile)


if __name__ == "__main__":
    main()
