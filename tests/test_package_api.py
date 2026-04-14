import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import praxis_bgm as praxis_pkg
import praxis_bgm.core as praxis_core
import praxis_bgm.utility as praxis_util


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _manual_logpdf(x, mu, cov):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    diff = x - mu
    inv = np.linalg.inv(cov)
    _, logdet = np.linalg.slogdet(cov)
    return -0.5 * (diff @ inv @ diff + logdet + x.shape[0] * np.log(2.0 * np.pi))


def _toy_cov(scale=1.0):
    base = np.array([[1.6, 0.2], [0.2, 0.9]], dtype=np.float32)
    return scale * base


@pytest.fixture(scope="module")
def toy_params():
    pis = jnp.array([0.55, 0.45], dtype=jnp.float32)
    mus = jnp.array([[0.0, 0.0], [1.0, -1.0]], dtype=jnp.float32)
    covs = jnp.array([_toy_cov(1.0), _toy_cov(1.2)], dtype=jnp.float32)
    Ls = praxis_util._prior_Ls_from_covs(covs)
    return praxis_util.MoGParams(pis, mus, Ls)


@pytest.fixture(scope="module")
def prior_components():
    prior_mus = jnp.array([[0.2, -0.1], [0.9, -0.8]], dtype=jnp.float32)
    prior_covs = jnp.array([_toy_cov(0.8), _toy_cov(1.1)], dtype=jnp.float32)
    prior_weights = jnp.array([0.5, 0.5], dtype=jnp.float32)
    prior_Ls = praxis_util._prior_Ls_from_covs(prior_covs)
    return prior_mus, prior_covs, prior_weights, prior_Ls


@pytest.fixture(scope="module")
def small_data():
    cluster_a = np.array(
        [
            [-2.2, -1.8],
            [-2.1, -1.9],
            [-1.8, -2.0],
            [-1.9, -1.6],
            [-2.0, -2.1],
            [-1.7, -1.8],
        ],
        dtype=np.float32,
    )
    cluster_b = np.array(
        [
            [1.8, 2.0],
            [2.0, 1.9],
            [2.1, 2.2],
            [1.9, 1.7],
            [2.2, 1.8],
            [1.7, 2.1],
        ],
        dtype=np.float32,
    )
    return np.vstack([cluster_a, cluster_b])


@pytest.fixture()
def fitted_model(small_data):
    model = praxis_core.Praxis_BGM(
        rng_key=jax.random.PRNGKey(0),
        K=2,
        verbose=False,
        prior_mus=np.array([[-1.5, -1.5], [1.5, 1.5]], dtype=np.float32),
        prior_Sigmas=np.stack([np.eye(2), np.eye(2)]).astype(np.float32),
        prior_weights=np.array([0.5, 0.5], dtype=np.float32),
        beta=0.001,
        rho_prec=0.1,
        rho_mu=1.0,
        num_samples=5,
        elbo_eval_freq=1,
        data_precision_int=1,
    )
    model.fit(small_data, num_iters=2, batch_size=4, early_stop=False, patience=2)
    return model


def test_package_exports_main_objects():
    assert praxis_pkg.Praxis_BGM is praxis_core.Praxis_BGM
    assert "build_gaussian_priors_from_source" in praxis_pkg.__all__
    assert "prepare_source_target_datasets" in praxis_pkg.__all__


@pytest.mark.parametrize(
    "diag_values, eps, expected",
    [
        ([-1.0, 2.0], 1e-6, [1e-6, 2.0]),
        ([0.0, 0.0], 0.1, [0.1, 0.1]),
        ([3.0, 4.0], 1e-6, [3.0, 4.0]),
    ],
)
def test_clip_diagonal_clips_only_diagonal(diag_values, eps, expected):
    L = jnp.array([[diag_values[0], -4.0], [3.0, diag_values[1]]], dtype=jnp.float32)
    clipped = np.asarray(praxis_util._clip_diagonal(L, eps=eps))
    np.testing.assert_allclose(np.diag(clipped), np.array(expected), atol=1e-6)
    assert clipped[0, 1] == pytest.approx(-4.0)
    assert clipped[1, 0] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "vec, K, message",
    [
        (np.array([1.0, 2.0, 3.0], dtype=np.float32), 2, "shape"),
        (np.array([1.0, np.nan], dtype=np.float32), 2, "finite"),
        (np.array([-1.0, 2.0], dtype=np.float32), 2, "nonnegative"),
        (np.array([0.0, 0.0], dtype=np.float32), 2, "positive"),
    ],
)
def test_validate_prob_vector_rejects_bad_inputs(vec, K, message):
    with pytest.raises(ValueError, match=message):
        praxis_util._validate_prob_vector("p", vec, K)


@pytest.mark.parametrize(
    "x, mu, cov",
    [
        (np.array([0.0, 0.0]), np.array([0.0, 0.0]), _toy_cov(1.0)),
        (np.array([1.0, -1.0]), np.array([0.5, 0.25]), _toy_cov(0.9)),
    ],
)
def test_logpdf_gaussian_matches_manual_formula(x, mu, cov):
    L = np.linalg.cholesky(cov).astype(np.float32)
    got = float(praxis_util.logpdf_gaussian(jnp.array(x), jnp.array(mu), jnp.array(L)))
    expected = _manual_logpdf(x, mu, cov)
    assert got == pytest.approx(expected, abs=1e-5)


def test_responsibilities_rows_sum_to_one(toy_params):
    data = np.array([[0.0, 0.0], [1.0, -1.0], [-0.3, 0.2]], dtype=np.float32)
    gamma = np.asarray(praxis_util.responsibilities(jnp.array(data), toy_params))
    np.testing.assert_allclose(gamma.sum(axis=1), np.ones(data.shape[0]), atol=1e-6)
    assert np.all(gamma >= 0.0)


@pytest.mark.parametrize(
    "pis",
    [
        np.array([0.5, 0.5], dtype=np.float32),
        np.array([0.8, 0.2], dtype=np.float32),
        np.array([0.2, 0.3, 0.5], dtype=np.float32),
    ],
)
def test_weights_eta_roundtrip(pis):
    eta = np.asarray(praxis_util.weights_to_eta(jnp.array(pis)))
    recovered = np.asarray(praxis_util.eta_to_weights(jnp.array(eta)))
    np.testing.assert_allclose(recovered, pis, atol=1e-6)


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.6])
def test_precision_and_covariance_chol_roundtrip(scale):
    cov = _toy_cov(scale)
    L = jnp.array(np.linalg.cholesky(cov).astype(np.float32))
    prec = np.asarray(praxis_util._precision_from_chol(L))
    recon_L = np.asarray(praxis_util._cov_chol_from_precision(jnp.array(prec)))
    recon_cov = recon_L @ recon_L.T
    np.testing.assert_allclose(recon_cov, cov, atol=1e-4)


def test_log_mog_grad_hess_matches_autodiff(toy_params):
    z = jnp.array([0.1, -0.2], dtype=jnp.float32)
    pis, mus, Ls = toy_params

    def scalar_fn(x):
        return praxis_util._log_mog(x, pis, mus, Ls)

    logq, grad, Hess = praxis_util._log_mog_grad_hess(z, pis, mus, Ls)
    grad_ad = np.asarray(jax.grad(scalar_fn)(z))
    hess_ad = np.asarray(jax.hessian(scalar_fn)(z))
    assert np.isfinite(float(logq))
    np.testing.assert_allclose(np.asarray(grad), grad_ad, atol=1e-5)
    np.testing.assert_allclose(np.asarray(Hess), hess_ad, atol=1e-5)


def test_compute_elbo_terms_returns_finite_outputs(toy_params, prior_components, small_data):
    prior_mus, prior_covs, prior_weights, prior_Ls = prior_components
    obs_precision = jnp.eye(2, dtype=jnp.float32)
    ll_data, reg, elbo = praxis_util.compute_elbo_terms(
        jnp.array(small_data),
        toy_params,
        prior_mus,
        prior_covs,
        obs_precision,
        jax.random.PRNGKey(0),
        prior_weights=prior_weights,
        prior_Ls=prior_Ls,
        tau=1.0,
        num_samples=4,
    )
    assert math.isfinite(float(ll_data))
    assert math.isfinite(float(reg))
    assert float(elbo) == pytest.approx(float(ll_data - reg), abs=1e-6)


def test_model_fit_predict_posteriors_and_summary(fitted_model, small_data):
    assignments, weights = fitted_model.predict(small_data)
    posterior_mus, posterior_covs, posterior_pis, responsibilities = fitted_model.get_posteriors(small_data)
    summary = fitted_model.get_model_summary()

    assert assignments.shape == (small_data.shape[0],)
    assert weights.shape == (2,)
    assert posterior_mus.shape == (2, 2)
    assert posterior_covs.shape == (2, 2, 2)
    assert posterior_pis.shape == (2,)
    assert responsibilities.shape == (small_data.shape[0], 2)
    assert summary["fitted"] is True
    assert summary["hyperparameters"]["K"] == 2
    assert np.isfinite(summary["learned_state"]["last_mc_elbo"])


def test_model_rejects_invalid_runtime_inputs(small_data):
    model = praxis_core.Praxis_BGM(
        rng_key=jax.random.PRNGKey(0),
        K=2,
        verbose=False,
        prior_mus=np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32),
        prior_Sigmas=np.stack([np.eye(2), np.eye(2)]).astype(np.float32),
        data_precision_int=1,
    )

    with pytest.raises(ValueError, match="2D array"):
        model.fit(np.array([1.0, 2.0], dtype=np.float32))

    bad = np.array(small_data, copy=True)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(bad)


def test_freeze_zero_mask_keeps_masked_covariances_zero(small_data):
    prior_covs = np.array([_toy_cov(1.0), _toy_cov(1.2)], dtype=np.float32)
    model = praxis_core.Praxis_BGM(
        rng_key=jax.random.PRNGKey(0),
        K=2,
        prior_mus=np.array([[-1.0, -1.0], [1.0, 1.0]], dtype=np.float32),
        prior_Sigmas=prior_covs,
        sparse_A=np.eye(2, dtype=np.float32),
        freeze_A_zeros=True,
        verbose=False,
        data_precision_int=1,
    )

    data_jnp = jnp.array(small_data)
    model._init_priors(data_jnp)
    model._init_params(data_jnp)
    init_covs = np.asarray(jax.vmap(lambda L: L @ L.T)(model.params.Ls))
    np.testing.assert_allclose(init_covs[:, 0, 1], np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(init_covs[:, 1, 0], np.zeros(2), atol=1e-6)
