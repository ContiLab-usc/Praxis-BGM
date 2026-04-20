import numpy as np
import pytest

import Praxis_tutorial as tutorial_module
import praxis_bgm.core as praxis_core

serial_module = pytest.importorskip("praxis_in_serial")


class _DummyBGM:
    def __init__(self, n_components, covariance_type, max_iter, random_state):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        center = X.mean(axis=0)
        offsets = np.linspace(-0.25, 0.25, self.n_components, dtype=np.float32)[:, None]
        self.means_ = center[None, :] + offsets
        return self


class _DummyGaussianMixture:
    def __init__(self, n_components, covariance_type, random_state, init_params, max_iter, n_init):
        self.n_components = n_components

    def fit(self, X):
        self._X = np.asarray(X, dtype=np.float32)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.arange(X.shape[0]) % self.n_components


def test_tutorial_run_tiny_profile(monkeypatch):
    tiny_profile = {
        "profile_name": "tiny",
        "profile_description": "Tiny smoke test profile.",
        "simulation_config": {
            "seed": 0,
            "K": 2,
            "d": 4,
            "N_src": 20,
            "N_tgt": 10,
            "source_mixture_weights": [0.7, 0.3],
            "target_mixture_weights": [0.3, 0.7],
            "frac_feat_shift": 0.25,
            "global_shift_scale": 0.1,
            "feat_shift_scale": 0.1,
            "cov_scale": 1.02,
            "flip_frac": 0.0,
        },
        "source_hparams": {
            "beta": 1e-3,
            "tol": 1e-4,
            "max_iters": 8,
            "num_iters": 2,
            "batch_size": 10,
            "num_samples": 4,
            "prior_mus_variance": 2.0,
            "likelihood_temp": 1.0,
            "rho_prec": 0.08,
            "rho_mu": 1.0,
            "elbo_eval_freq": 1,
            "data_precision_int": 1,
            "early_stop": False,
            "patience": 2,
        },
        "target_transfer_hparams": {
            "beta": 1e-3,
            "tol": 1e-4,
            "max_iters": 8,
            "num_iters": 2,
            "batch_size": 10,
            "num_samples": 4,
            "prior_mus_variance": 1.0,
            "likelihood_temp": 1.0,
            "rho_prec": 0.08,
            "rho_mu": 1.0,
            "elbo_eval_freq": 1,
            "data_precision_int": 1,
            "early_stop": False,
            "patience": 2,
            "covariance_transfer_scale": 1.02,
        },
        "target_baseline_hparams": {
            "beta": 1e-3,
            "tol": 1e-4,
            "max_iters": 8,
            "num_iters": 2,
            "batch_size": 10,
            "num_samples": 4,
            "prior_mus_variance": 2.0,
            "likelihood_temp": 1.0,
            "rho_prec": 0.08,
            "rho_mu": 1.0,
            "elbo_eval_freq": 1,
            "data_precision_int": 1,
            "early_stop": False,
            "patience": 2,
        },
    }

    monkeypatch.setattr(tutorial_module, "get_tutorial_profile", lambda _: tiny_profile)
    monkeypatch.setattr(praxis_core, "BayesianGaussianMixture", _DummyBGM)
    results = tutorial_module.run_tutorial("tiny")

    assert results["profile_name"] == "tiny"
    assert results["shifted_feature_count"] >= 1
    assert "ari_target_with_transferred_priors" in results
    assert "ari_target_without_priors" in results


def test_serial_helper_functions_work_on_small_inputs(monkeypatch):
    Z1, Z2, Z3, pi, T12, T23 = serial_module.simulate_three_layer_clusters(N=15, K=3, seed=1)
    assert Z1.shape == (15,)
    assert Z2.shape == (15,)
    assert Z3.shape == (15,)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(T12.sum(axis=1), np.ones(3), atol=1e-6)
    np.testing.assert_allclose(T23.sum(axis=1), np.ones(3), atol=1e-6)

    rng = np.random.default_rng(0)
    mus, Sigmas = serial_module.simulate_layer_params(3, 6, rng)
    X = serial_module.simulate_layer_data(Z1, mus, Sigmas, rng)
    mus_prior, Sigmas_prior = serial_module.compute_clusterwise_moments(X, Z1, 3)
    monkeypatch.setattr(serial_module, "GaussianMixture", _DummyGaussianMixture)
    bgm_labels = serial_module.fit_bgm_baseline(X, 3, seed=0)

    assert X.shape == (15, 6)
    assert mus_prior.shape == (3, 6)
    assert Sigmas_prior.shape == (3, 6, 6)
    assert bgm_labels.shape == (15,)


def test_serial_fit_praxis_layer_returns_model_and_assignments(monkeypatch):
    X = np.array(
        [
            [-2.0, -2.1],
            [-1.8, -1.9],
            [-2.2, -1.7],
            [2.0, 2.1],
            [1.9, 1.8],
            [2.2, 1.7],
        ],
        dtype=np.float32,
    )

    monkeypatch.setattr(praxis_core, "BayesianGaussianMixture", _DummyBGM)
    model, labels, weights = serial_module.fit_praxis_layer(
        serial_module.PRNGKey(0),
        X,
        K=2,
        beta=1e-3,
    )

    assert model.params is not None
    assert labels.shape == (6,)
    assert weights.shape == (2,)
