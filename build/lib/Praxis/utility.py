"""Numerical utilities for the packaged Praxis implementation.

This module contains the low-level linear algebra, density evaluation,
Monte-Carlo ELBO estimation, and damped NGD/VON update routines used by the
high-level :class:`Praxis_BGM` API in ``core.py``.

Most helpers are intentionally kept close to the original research/prototype
implementation so the packaged library preserves the behavior of
``Praxis_BGM_global_z_prior_damped.py`` while separating infrastructure from the
user-facing model class.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial
from jax import jit, vmap
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
import numpy as np
from typing import NamedTuple

_DEFAULT = object()

__all__ = [
    "_DEFAULT",
    "_clip_diagonal",
    "batched_cov_to_decay_scaled",
    "single_cov_to_decay_scaled",
    "_symmetrize",
    "_psd_clip",
    "_stabilize_covariance",
    "_stabilize_covariances",
    "_prepare_zero_freeze_mask",
    "_validate_prob_vector",
    "_validate_means",
    "_validate_covariances",
    "MoGParams",
    "_params_are_finite",
    "logpdf_gaussian",
    "vectorized_logpdf",
    "responsibilities",
    "weights_to_eta",
    "eta_to_weights",
    "_phi_lower",
    "_clip_tril_positive",
    "_chol_PD",
    "_precision_from_chol",
    "_cov_chol_from_precision",
    "_freeze_zero_masked_covariance",
    "_cov_from_precision",
    "_chol_from_masked_precision",
    "_chol_from_masked_covariance",
    "_log_normal_at_point",
    "_log_mog",
    "_prior_Ls_from_covs",
    "_log_prior_mog",
    "_log_mog_grad_hess",
    "_delta_at_z0",
    "_batch_loglik_anchor",
    "_batch_loglik_grad_hess",
    "_h_strict",
    "_dataset_loglik_at_z",
    "compute_elbo_terms",
    "compute_elbo",
    "_anchor_statistics",
    "ngd_update",
    "normal_pdf",
    "compute_bf_for_cluster",
    "compute_bf_matrix",
    "compute_feature_scores_from_bf",
    "compute_bayes_factors_and_scores",
]


@jax.jit
def _clip_diagonal(L, eps=1e-9):
    """Clip only the diagonal of a square matrix to be at least ``eps``."""
    d = L.shape[-1]
    diag_vals = jnp.diag(L)
    clipped_diag = jnp.clip(diag_vals, min=eps)
    return L.at[jnp.diag_indices(d)].set(clipped_diag)


@jax.jit
def batched_cov_to_decay_scaled(cov_matrices: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """Convert a batch of covariance matrices to scaled decay masks.

    The transform first converts covariance to correlation, then maps absolute
    correlations into ``(0, 1]`` weights using the decay rule from the original
    Praxis experiments.
    """

    def cov2scaled_decay(cov):
        stds = jnp.sqrt(jnp.diag(cov) + eps)
        D = jnp.diag(1.0 / stds)
        corr = D @ cov @ D
        corr = corr.at[jnp.diag_indices(corr.shape[0])].set(1.0)
        mag = jnp.abs(corr)
        raw = jnp.exp(-jnp.square(1.0 - mag))
        return raw.at[jnp.diag_indices(raw.shape[0])].set(1.0)

    return jax.vmap(cov2scaled_decay)(cov_matrices)


@jax.jit
def single_cov_to_decay_scaled(cov_matrix: jnp.ndarray, eps: float = 1e-9) -> jnp.ndarray:
    """Single-matrix version of :func:`batched_cov_to_decay_scaled`."""
    stds = jnp.sqrt(jnp.diag(cov_matrix) + eps)
    D = jnp.diag(1.0 / stds)
    corr = D @ cov_matrix @ D
    corr = corr.at[jnp.diag_indices(corr.shape[0])].set(1.0)
    mag = jnp.abs(corr)
    raw = jnp.exp(-jnp.square(1.0 - mag))
    return raw.at[jnp.diag_indices(raw.shape[0])].set(1.0)


@jax.jit
def _symmetrize(M):
    """Return the symmetric part ``(M + M^T) / 2`` of a matrix."""
    return 0.5 * (M + M.T)


@jax.jit
def _psd_clip(M, eps=1e-8):
    """Project a symmetric matrix onto the PSD cone by eigenvalue clipping."""
    M = _symmetrize(M)
    evals, evecs = jnp.linalg.eigh(M)
    evals = jnp.clip(evals, min=eps)
    return (evecs * evals) @ evecs.T


@jax.jit
def _stabilize_covariance(cov, eps=1e-8):
    """Symmetrize and clip a covariance matrix to be positive semidefinite."""
    return _psd_clip(_symmetrize(cov), eps=eps)


@jax.jit
def _stabilize_covariances(covs, eps=1e-8):
    """Apply :func:`_stabilize_covariance` to a batch of covariance matrices."""
    return vmap(lambda cov: _stabilize_covariance(cov, eps))(covs)


def _prepare_zero_freeze_mask(K, d, sparse_A=None, cluster_A=None):
    """Build the training mask used when ``freeze_A_zeros=True``.

    Parameters
    ----------
    K:
        Number of mixture components.
    d:
        Feature dimension.
    sparse_A:
        Optional global binary mask of shape ``(d, d)``.
    cluster_A:
        Optional per-cluster binary mask of shape ``(K, d, d)``.

    Returns
    -------
    jax.Array or None
        A symmetric binary mask with shape ``(K, d, d)`` where zeros denote
        covariance entries that should be frozen to zero throughout training.
    """
    if (sparse_A is None) and (cluster_A is None):
        return None

    allowed = np.ones((K, d, d), dtype=bool)
    if sparse_A is not None:
        sparse_arr = np.asarray(sparse_A)
        if sparse_arr.shape != (d, d):
            raise ValueError(f"sparse_A must have shape ({d}, {d}), got {sparse_arr.shape}.")
        allowed &= (sparse_arr > 0)[None, :, :]

    if cluster_A is not None:
        cluster_arr = np.asarray(cluster_A)
        if cluster_arr.shape != (K, d, d):
            raise ValueError(f"cluster_A must have shape ({K}, {d}, {d}), got {cluster_arr.shape}.")
        allowed &= (cluster_arr > 0)

    allowed &= np.swapaxes(allowed, -1, -2)
    for k in range(K):
        np.fill_diagonal(allowed[k], True)

    return jnp.array(allowed.astype(np.float32))


def _validate_prob_vector(name, vec, K):
    """Validate and normalize a length-``K`` nonnegative probability vector."""
    arr = np.asarray(vec, dtype=np.float32)
    if arr.shape != (K,):
        raise ValueError(f"{name} must have shape ({K},), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be nonnegative.")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError(f"{name} must sum to a positive value.")
    return arr / total


def _validate_means(name, mus, K):
    """Validate a matrix of prior or initial means with shape ``(K, d)``."""
    arr = np.asarray(mus, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array of shape ({K}, d).")
    if arr.shape[0] != K:
        raise ValueError(f"{name} must have first dimension {K}, got {arr.shape[0]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _validate_covariances(name, covs, K, d=None):
    """Validate a stack of covariance matrices with shape ``(K, d, d)``."""
    arr = np.asarray(covs, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3D array of shape ({K}, d, d).")
    if arr.shape[0] != K:
        raise ValueError(f"{name} must have first dimension {K}, got {arr.shape[0]}.")
    if arr.shape[1] != arr.shape[2]:
        raise ValueError(f"{name} must contain square covariance matrices, got {arr.shape[1:]}.")
    if (d is not None) and (arr.shape[1] != d):
        raise ValueError(f"{name} must have trailing dimensions ({d}, {d}), got {arr.shape[1:]}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


class MoGParams(NamedTuple):
    """Packed variational mixture parameters.

    Attributes
    ----------
    pis:
        Mixture weights with shape ``(K,)``.
    mus:
        Component means with shape ``(K, d)``.
    Ls:
        Lower-triangular Cholesky factors with shape ``(K, d, d)`` where
        ``Sigma_k = L_k L_k^T``.
    """

    pis: jnp.ndarray
    mus: jnp.ndarray
    Ls: jnp.ndarray


@jit
def _params_are_finite(params: MoGParams):
    """Return ``True`` when all packed parameters are finite."""
    return (
        jnp.all(jnp.isfinite(params.pis))
        & jnp.all(jnp.isfinite(params.mus))
        & jnp.all(jnp.isfinite(params.Ls))
    )


@jit
def logpdf_gaussian(x, mu, L):
    """Compute ``log N(x | mu, L L^T)`` without explicitly inverting covariance."""
    L_stable = _clip_diagonal(L)
    d = x.shape[-1]
    diff = x - mu
    z = jnp.linalg.solve(L_stable, diff)
    quad_form = jnp.dot(z, z)
    logdet_Sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(L_stable) + 1e-12))
    return -0.5 * (quad_form + logdet_Sigma + d * jnp.log(2.0 * jnp.pi))


@jit
def vectorized_logpdf(data, mus, Ls):
    """Evaluate each observation against each mixture component.

    Returns an array with shape ``(N, K)``.
    """
    log_probs = vmap(lambda k: vmap(lambda x: logpdf_gaussian(x, mus[k], Ls[k]))(data))(jnp.arange(mus.shape[0]))
    return log_probs.T


@jit
def responsibilities(data, params: MoGParams):
    """Compute posterior cluster responsibilities for each observation."""
    pis, mus, Ls = params
    log_mix = jnp.log(pis + 1e-12)[None, :] + vectorized_logpdf(data, mus, Ls)
    log_norm = logsumexp(log_mix, axis=1, keepdims=True)
    return jnp.exp(log_mix - log_norm)


@jit
def weights_to_eta(pis):
    """Map simplex weights to unconstrained ``K-1`` softmax logits."""
    eps = 1e-12
    ref = pis[-1] + eps
    return jnp.log(pis[:-1] + eps) - jnp.log(ref)


@jit
def eta_to_weights(eta):
    """Map unconstrained ``K-1`` logits back to a ``K``-simplex probability vector."""
    eta_full = jnp.concatenate([eta, jnp.array([0.0], dtype=eta.dtype)])
    return jax.nn.softmax(eta_full)


@jit
def _phi_lower(X: jnp.ndarray) -> jnp.ndarray:
    """Keep the lower triangle of ``X`` and halve the diagonal."""
    diag = jnp.diag(jnp.diag(X)) * 0.5
    return jnp.tril(X, -1) + diag


@jit
def _clip_tril_positive(L, eps=1e-6):
    """Force a matrix to be lower triangular with strictly positive diagonal."""
    L = jnp.tril(L)
    diag = jnp.clip(jnp.diag(L), min=eps)
    return L - jnp.diag(jnp.diag(L)) + jnp.diag(diag)


@jit
def _chol_PD(L):
    """Stabilize a candidate Cholesky factor into a valid lower-triangular factor."""
    return _clip_tril_positive(jnp.tril(L))


@jit
def _precision_from_chol(L):
    """Convert a covariance Cholesky factor into a precision matrix."""
    d = L.shape[-1]
    I = jnp.eye(d, dtype=L.dtype)
    Ls = _chol_PD(L)
    Linv = jsp.linalg.solve_triangular(Ls, I, lower=True)
    return jsp.linalg.solve_triangular(Ls.T, Linv, lower=False)


@jit
def _cov_chol_from_precision(Prec, eps=1e-8):
    """Recover a covariance Cholesky factor from a precision matrix."""
    Prec = _psd_clip(_symmetrize(Prec), eps=eps)
    d = Prec.shape[-1]
    I = jnp.eye(d, dtype=Prec.dtype)
    chol_prec = jsp.linalg.cholesky(Prec + eps * I, lower=True)
    Sigma = jsp.linalg.cho_solve((chol_prec, True), I)
    Sigma = _stabilize_covariance(Sigma, eps=eps)
    return jsp.linalg.cholesky(Sigma + eps * I, lower=True)


@jit
def _freeze_zero_masked_covariance(cov, A_mask, eps=1e-8):
    """Zero masked covariance entries while preserving positive definiteness."""
    cov = _symmetrize(cov)
    d = cov.shape[-1]
    I = jnp.eye(d, dtype=cov.dtype)
    off_mask = A_mask * (1.0 - I)
    off = cov * off_mask
    row_abs = jnp.sum(jnp.abs(off), axis=1)
    diag_new = jnp.maximum(jnp.diag(cov), row_abs + eps)
    return off + jnp.diag(diag_new)


@jit
def _cov_from_precision(Prec, eps=1e-8):
    """Compute covariance by inverting a stabilized precision matrix."""
    Prec = _psd_clip(_symmetrize(Prec), eps=eps)
    d = Prec.shape[-1]
    I = jnp.eye(d, dtype=Prec.dtype)
    chol_prec = jsp.linalg.cholesky(Prec + eps * I, lower=True)
    return jsp.linalg.cho_solve((chol_prec, True), I)


@jit
def _chol_from_masked_precision(Prec, A_mask, eps=1e-8):
    """Convert precision to covariance, apply a zero mask, then re-factorize."""
    Sigma = _cov_from_precision(Prec, eps=eps)
    Sigma_masked = _freeze_zero_masked_covariance(Sigma, A_mask, eps=eps)
    d = Sigma_masked.shape[-1]
    I = jnp.eye(d, dtype=Sigma_masked.dtype)
    return jsp.linalg.cholesky(Sigma_masked + eps * I, lower=True)


@jit
def _chol_from_masked_covariance(cov, A_mask, eps=1e-8):
    """Apply a zero mask directly in covariance space and return its Cholesky factor."""
    Sigma_masked = _freeze_zero_masked_covariance(cov, A_mask, eps=eps)
    d = Sigma_masked.shape[-1]
    I = jnp.eye(d, dtype=Sigma_masked.dtype)
    return jsp.linalg.cholesky(Sigma_masked + eps * I, lower=True)


@jax.jit
def _log_normal_at_point(z0, mu, L):
    """Evaluate a Gaussian component at a single anchor point."""
    return logpdf_gaussian(z0, mu, _chol_PD(L))


@jax.jit
def _log_mog(z, pis, mus, Ls):
    """Log density of the current variational Gaussian mixture ``q(z)``."""
    Ls_chol = vmap(_chol_PD)(Ls)
    log_comps = jnp.log(pis + 1e-12) + vmap(logpdf_gaussian, (None, 0, 0))(z, mus, Ls_chol)
    return jsp.special.logsumexp(log_comps)


@jax.jit
def _prior_Ls_from_covs(prior_Sigmas):
    """Convert prior covariance matrices into stabilized Cholesky factors."""
    def cholS(S):
        S = _stabilize_covariance(S)
        d = S.shape[-1]
        return jsp.linalg.cholesky(S + 1e-9 * jnp.eye(d, dtype=S.dtype), lower=True)

    return vmap(cholS)(prior_Sigmas)


@jax.jit
def _log_prior_mog(z, pi0, prior_mus, prior_Sigmas=None, prior_Ls=None):
    """Log density of the Gaussian-mixture prior over latent anchors ``z``."""
    if prior_Ls is None:
        prior_Ls = _prior_Ls_from_covs(prior_Sigmas)
    log_comps = jnp.log(pi0 + 1e-12) + vmap(logpdf_gaussian, (None, 0, 0))(z, prior_mus, prior_Ls)
    return jsp.special.logsumexp(log_comps)


@jax.jit
def _log_mog_grad_hess(z, pis, mus, Ls):
    """Return ``log q(z)``, its gradient, and Hessian under the mixture model."""
    Ls = vmap(_chol_PD)(Ls)

    def comp_stats(mu, L):
        Prec = _precision_from_chol(L)
        diff = z - mu
        score = -(Prec @ diff)
        Hlog = -Prec
        logN = logpdf_gaussian(z, mu, L)
        return logN, score, Hlog

    logN, scores, Hlogs = vmap(comp_stats)(mus, Ls)
    logw = jnp.log(pis + 1e-12) + logN
    logq = jsp.special.logsumexp(logw)
    r = jax.nn.softmax(logw)

    grad = jnp.sum(r[:, None] * scores, axis=0)
    second_moment = jnp.sum(
        r[:, None, None] * (Hlogs + jnp.einsum("ki,kj->kij", scores, scores)),
        axis=0,
    )
    Hess = second_moment - jnp.outer(grad, grad)
    return logq, grad, _symmetrize(Hess)


@jax.jit
def _delta_at_z0(z0, pis, mus, Ls):
    """Compute the anchor coupling weights ``delta_j(z0)`` for one sampled anchor."""
    Ls = vmap(_chol_PD)(Ls)
    logN = vmap(lambda m, L: _log_normal_at_point(z0, m, L))(mus, Ls)
    log_den = jsp.special.logsumexp(jnp.log(pis + 1e-12) + logN)
    return jnp.exp(logN - log_den)


@jax.jit
def _batch_loglik_anchor(batch, z0, obs_precision=None, N_total=None, tau=1.0):
    """Scaled minibatch approximation to the observation log-likelihood term."""
    if (batch is None) or (obs_precision is None) or (N_total is None):
        return 0.0

    M = batch.shape[0]
    scale_mb = tau * (N_total / M)
    d = z0.shape[0]
    diff = batch - z0
    quad = jnp.einsum("ni,ij,nj->n", diff, obs_precision, diff)
    sign, logdet_precision = jnp.linalg.slogdet(obs_precision)
    logdet_precision = jnp.where(sign > 0, logdet_precision, -jnp.inf)
    loglikes = 0.5 * (logdet_precision - quad - d * jnp.log(2.0 * jnp.pi))
    return scale_mb * jnp.mean(loglikes)


@jax.jit
def _batch_loglik_grad_hess(batch, z0, obs_precision=None, N_total=None, tau=1.0):
    """Gradient and Hessian of the negative scaled minibatch likelihood term."""
    d = z0.shape[0]
    if (batch is None) or (obs_precision is None) or (N_total is None):
        return jnp.zeros((d,), dtype=z0.dtype), jnp.zeros((d, d), dtype=z0.dtype)

    M = batch.shape[0]
    scale_mb = tau * (N_total / M)
    barD = jnp.mean(batch, axis=0)
    g = scale_mb * (obs_precision @ (z0 - barD))
    H = scale_mb * obs_precision
    return g, H


@jax.jit
def _h_strict(
    z0,
    batch,
    pis,
    mus,
    Ls,
    prior_mus,
    prior_Sigmas,
    prior_Ls=None,
    pi0=None,
    obs_precision=None,
    N_total=None,
    tau=1.0,
):
    """Shared scalar objective evaluated at an anchor sample ``z0``.

    This is the damped global-z-prior objective

    ``h(z0) = log q(z0) - log p(z0) - scaled_batch_loglik(z0)``.
    """
    K = pis.shape[0]
    if pi0 is None:
        pi0 = jnp.full((K,), 1.0 / K, dtype=pis.dtype)

    log_q = _log_mog(z0, pis, mus, Ls)
    log_p = _log_prior_mog(z0, pi0, prior_mus, prior_Sigmas=prior_Sigmas, prior_Ls=prior_Ls)
    avg_loglike = _batch_loglik_anchor(
        batch,
        z0,
        obs_precision=obs_precision,
        N_total=N_total,
        tau=tau,
    )
    return (log_q - log_p) - avg_loglike


@jax.jit
def _dataset_loglik_at_z(data, z0, obs_precision, tau=1.0):
    """Full-data observation log-likelihood at a fixed latent anchor ``z0``."""
    d = z0.shape[0]
    diff = data - z0
    quad = jnp.einsum("ni,ij,nj->n", diff, obs_precision, diff)
    sign, logdet_precision = jnp.linalg.slogdet(obs_precision)
    logdet_precision = jnp.where(sign > 0, logdet_precision, -jnp.inf)
    loglikes = 0.5 * (logdet_precision - quad - d * jnp.log(2.0 * jnp.pi))
    return tau * jnp.sum(loglikes)


@partial(jax.jit, static_argnames=("num_samples",))
def compute_elbo_terms(
    data,
    params: MoGParams,
    prior_mus,
    prior_Sigmas,
    obs_precision,
    rng_key,
    prior_weights=None,
    prior_Ls=None,
    tau=1.0,
    num_samples=100,
):
    """Monte Carlo estimate of the ELBO decomposition.

    Returns the data term, regularization term, and their difference using
    anchors sampled from the current variational mixture.
    """
    pis, mus, Ls = params
    K, d = mus.shape

    if prior_weights is None:
        prior_weights = jnp.full((K,), 1.0 / K, dtype=pis.dtype)
    if prior_Ls is None:
        prior_Ls = _prior_Ls_from_covs(prior_Sigmas)

    rng_key, k_pi, k_eps = jax.random.split(rng_key, 3)
    i = jax.random.categorical(k_pi, jnp.log(pis + 1e-12), shape=(num_samples,))
    sampled_Ls = vmap(_chol_PD)(Ls[i])
    eps = jax.random.normal(k_eps, (num_samples, d))
    z_samples = mus[i] + vmap(lambda L, e: L @ e)(sampled_Ls, eps)

    ll_samples = vmap(lambda z: _dataset_loglik_at_z(data, z, obs_precision, tau=tau))(z_samples)
    logq_samples = vmap(lambda z: _log_mog(z, pis, mus, Ls))(z_samples)
    logp_samples = vmap(
        lambda z: _log_prior_mog(z, prior_weights, prior_mus, prior_Sigmas=prior_Sigmas, prior_Ls=prior_Ls)
    )(z_samples)

    ll_data = jnp.mean(ll_samples)
    reg = jnp.mean(logq_samples - logp_samples)
    return ll_data, reg, ll_data - reg


@partial(jax.jit, static_argnames=("num_samples",))
def compute_elbo(
    data,
    params: MoGParams,
    prior_mus,
    prior_Sigmas,
    obs_precision,
    rng_key,
    prior_weights=None,
    prior_Ls=None,
    tau=1.0,
    num_samples=100,
):
    """Convenience wrapper returning only the Monte Carlo ELBO estimate."""
    _, _, elbo = compute_elbo_terms(
        data,
        params,
        prior_mus,
        prior_Sigmas,
        obs_precision,
        rng_key,
        prior_weights=prior_weights,
        prior_Ls=prior_Ls,
        tau=tau,
        num_samples=num_samples,
    )
    return elbo


@jax.jit
def _anchor_statistics(
    z0,
    batch,
    pis,
    mus,
    Ls,
    prior_mus,
    prior_Sigmas,
    prior_weights,
    prior_Ls,
    obs_precision,
    N_total,
    tau,
):
    """Collect all anchor-dependent statistics needed for one update step."""
    deltas = _delta_at_z0(z0, pis, mus, Ls)
    h_scalar = _h_strict(
        z0,
        batch,
        pis,
        mus,
        Ls,
        prior_mus,
        prior_Sigmas,
        prior_Ls=prior_Ls,
        pi0=prior_weights,
        obs_precision=obs_precision,
        N_total=N_total,
        tau=tau,
    )
    _, g_q, H_q = _log_mog_grad_hess(z0, pis, mus, Ls)
    _, g_p, H_p = _log_mog_grad_hess(z0, prior_weights, prior_mus, prior_Ls)
    g_like, H_like = _batch_loglik_grad_hess(
        batch,
        z0,
        obs_precision=obs_precision,
        N_total=N_total,
        tau=tau,
    )
    grad_h = g_q - g_p + g_like
    Hess_h = _symmetrize(H_q - H_p + H_like)
    return deltas, h_scalar, grad_h, Hess_h


@partial(jax.jit, static_argnames=("num_samples", "freeze_A_zeros"))
def ngd_update(
    params: MoGParams,
    batch: jnp.ndarray,
    prior_mus: jnp.ndarray,
    prior_Sigmas: jnp.ndarray,
    beta: float,
    rng_key,
    prior_weights: jnp.ndarray = None,
    prior_Ls: jnp.ndarray = None,
    obs_precision: jnp.ndarray = None,
    N_total: int = None,
    tau: float = 1.0,
    num_samples: int = 100,
    A_zero_mask: jnp.ndarray = None,
    freeze_A_zeros: bool = False,
    rho_prec: float = 0.05,
    rho_mu: float = 1.0,
):
    """Perform one damped NGD/VON update on mixture weights, means, and covariances.

    Parameters
    ----------
    params:
        Current variational parameters.
    batch:
        Minibatch of observations with shape ``(M, d)``.
    prior_mus, prior_Sigmas, prior_weights, prior_Ls:
        Parameters of the Gaussian-mixture prior over latent anchors.
    beta:
        Step size used in the mixture-weight and mean updates.
    obs_precision:
        Observation precision used in ``p(x | z)``.
    N_total:
        Total dataset size for minibatch scaling.
    tau:
        Likelihood temperature / scaling factor.
    num_samples:
        Number of anchor samples drawn from the current mixture.
    A_zero_mask, freeze_A_zeros:
        Optional mask and flag used to enforce zero patterns in covariance.
    rho_prec:
        Damping coefficient for the precision update.
    rho_mu:
        Damping coefficient for the mean update.
    """
    pis, mus, Ls = params
    K, d = mus.shape

    if prior_weights is None:
        prior_weights = jnp.full((K,), 1.0 / K, dtype=pis.dtype)
    if prior_Ls is None:
        prior_Ls = _prior_Ls_from_covs(prior_Sigmas)

    num_samples = max(int(num_samples), 1)

    rng_key, k_pi, k_eps = jax.random.split(rng_key, 3)
    i = jax.random.categorical(k_pi, jnp.log(pis + 1e-12), shape=(num_samples,))
    sampled_Ls = vmap(_chol_PD)(Ls[i])
    eps = jax.random.normal(k_eps, (num_samples, d))
    z_samples = mus[i] + vmap(lambda L, e: L @ e)(sampled_Ls, eps)

    deltas, h_samples, grad_samples, Hess_samples = vmap(
        lambda z: _anchor_statistics(
            z,
            batch,
            pis,
            mus,
            Ls,
            prior_mus,
            prior_Sigmas,
            prior_weights,
            prior_Ls,
            obs_precision,
            N_total,
            tau,
        )
    )(z_samples)

    h_scalar = jnp.mean(h_samples)
    delta_grad = jnp.einsum("sk,sd->kd", deltas, grad_samples) / num_samples
    delta_Hess = jnp.einsum("sk,sab->kab", deltas, Hess_samples) / num_samples
    delta_Hess = vmap(_symmetrize)(delta_Hess)

    current_precisions = vmap(_precision_from_chol)(Ls)
    target_precisions = vmap(_psd_clip)(delta_Hess)
    new_precisions = (1.0 - rho_prec) * current_precisions + rho_prec * target_precisions
    if freeze_A_zeros:
        new_Ls = vmap(_chol_from_masked_precision)(new_precisions, A_zero_mask)
    else:
        new_Ls = vmap(_cov_chol_from_precision)(new_precisions)

    def mean_update(mu_j, L_new_j, grad_eff_j):
        Sigma_new_j = L_new_j @ L_new_j.T
        return mu_j - rho_mu * beta * (Sigma_new_j @ grad_eff_j)

    new_mus = vmap(mean_update)(mus, new_Ls, delta_grad)

    rho = weights_to_eta(pis)
    delta_eta = -beta * jnp.mean((deltas[:, :-1] - deltas[:, [-1]]) * h_samples[:, None], axis=0)
    rho_new = rho + delta_eta
    pis_new = eta_to_weights(rho_new)

    new_params = MoGParams(pis_new, new_mus, new_Ls)
    return new_params, rng_key, h_scalar


def normal_pdf(x, mean, var):
    """Univariate normal density helper used by Bayes factor utilities."""
    return norm.pdf(x, loc=mean, scale=jnp.sqrt(var))


def compute_bf_for_cluster(j, X, m, v, delta):
    """Compute per-feature Bayes factors for one cluster against a null model."""
    eps = 1e-10
    delta_j = delta[:, j]
    like_cluster = norm.pdf(X, loc=m[j, :], scale=jnp.sqrt(v[j, :]) + eps)
    null_mean = jnp.mean(X, axis=0)
    null_var = jnp.var(X, axis=0)
    like_null = norm.pdf(X, loc=null_mean, scale=jnp.sqrt(null_var) + eps)
    mix_like = delta_j[:, None] * like_cluster + (1 - delta_j)[:, None] * like_null
    ratio = mix_like / (like_null + eps)
    log_ratio = jnp.log(ratio + eps)
    sum_log_ratio = jnp.sum(log_ratio, axis=0)
    sum_log_ratio = jnp.clip(sum_log_ratio, min=-100.0, max=100.0)
    bf = jnp.exp(sum_log_ratio)
    return bf


def compute_bf_matrix(X, m, v, delta):
    """Compute Bayes-factor scores for all clusters and features."""
    BF_matrix = jax.vmap(lambda j: compute_bf_for_cluster(j, X, m, v, delta))(jnp.arange(m.shape[0]))
    return BF_matrix


def compute_feature_scores_from_bf(BF_matrix, theta):
    """Aggregate per-cluster Bayes factors into global feature scores."""
    eps = 1e-10
    BF_matrix_clipped = jnp.clip(BF_matrix, min=eps, max=1e12)
    weighted_log = jnp.sum(theta[:, None] * jnp.log(BF_matrix_clipped + eps), axis=0)
    weighted_log = jnp.clip(weighted_log, min=-100.0, max=100.0)
    score = jnp.exp(weighted_log)
    score = jnp.nan_to_num(score, nan=1.0, posinf=1e12, neginf=eps)
    return score


def compute_bayes_factors_and_scores(X, m, v, delta, theta):
    """Return both the Bayes-factor matrix and aggregated feature scores."""
    BF_matrix = compute_bf_matrix(X, m, v, delta)
    feature_scores = compute_feature_scores_from_bf(BF_matrix, theta)
    return BF_matrix, feature_scores
