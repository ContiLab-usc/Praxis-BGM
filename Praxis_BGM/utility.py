# ==============================================================================
# Praxis-BGM with VON + Structural A-Mask on Precision/Covariance
# ==============================================================================

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.random import split, permutation
import jax.scipy as jsp
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm

import jax.numpy as jnp
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from typing import NamedTuple
from functools import partial
import time

# ------------------------------------------------------------------------------
#                       Named tuple for mixture params
# ------------------------------------------------------------------------------
class MoGParams(NamedTuple):
    pis: jnp.ndarray        # (K,)
    mus: jnp.ndarray        # (K, d)
    Lambdas: jnp.ndarray    # (K, d, d)

__all__ = [
    "MoGParams",
    "logpdf_gaussian_prec",
    "vectorized_logpdf_prec",
    "responsibilities",
    "gaussian_kl",
    "compute_elbo",
    "weights_to_eta",
    "eta_to_weights",
    "_log_mog",
    "_log_prior_mog",
    "_delta_at_z0",
    "_batch_loglik_anchor",
    "_h_strict",
    "_precision_from_cov",
    "_sample_from_precision",
    "_prepare_A_mask_global_or_cluster",
    "_apply_precision_mask_spd",
    "_apply_covariance_mask_from_precision",
    "_mask_all_components",
    "ngd_update_multi",
    "normal_pdf",
    "compute_bf_for_cluster",
    "compute_bf_matrix",
    "compute_feature_scores_from_bf",
    "compute_bayes_factors_and_scores",
]

# ------------------------------------------------------------------------------
#                          Basic Gaussian (precision form)
# ------------------------------------------------------------------------------
@jit
def logpdf_gaussian_prec(x: jnp.ndarray, mu: jnp.ndarray, Lambda: jnp.ndarray) -> jnp.ndarray:
    d = x.shape[-1]
    diff = x - mu
    quad = diff @ (Lambda @ diff)
    sign, logdet_L = jnp.linalg.slogdet(Lambda)
    # If numerical issue, add tiny jitter and recompute
    logdet_L = jnp.where(sign > 0, logdet_L,
                         jnp.linalg.slogdet(Lambda + 1e-9*jnp.eye(d, dtype=Lambda.dtype))[1])
    return -0.5 * (quad - logdet_L + d * jnp.log(2.0 * jnp.pi))

@jit
def vectorized_logpdf_prec(data: jnp.ndarray, mus: jnp.ndarray, Lambdas: jnp.ndarray) -> jnp.ndarray:
    return vmap(lambda x: vmap(lambda m, La: logpdf_gaussian_prec(x, m, La))(mus, Lambdas))(data)

# ------------------------------------------------------------------------------
#                       Responsibilities
# ------------------------------------------------------------------------------
@jit
def responsibilities(data: jnp.ndarray, params: MoGParams) -> jnp.ndarray:
    pis, mus, Lambdas = params
    log_mix = jnp.log(pis + 1e-12)[None, :] + vectorized_logpdf_prec(data, mus, Lambdas)
    log_norm = logsumexp(log_mix, axis=1, keepdims=True)
    return jnp.exp(log_mix - log_norm)

# ------------------------------------------------------------------------------
#                    Monitoring utilities (KL/ELBO)
# ------------------------------------------------------------------------------
@jit
def gaussian_kl(mu: jnp.ndarray, Sigma: jnp.ndarray, m0: jnp.ndarray, S0: jnp.ndarray) -> jnp.ndarray:
    d = mu.shape[0]
    I = jnp.eye(d, dtype=Sigma.dtype)
    S0_inv = jnp.linalg.solve(S0 + 1e-9*I, I)
    logdet_S0 = jnp.linalg.slogdet(S0 + 1e-9*I)[1]
    logdet_S = jnp.linalg.slogdet(Sigma + 1e-9*I)[1]
    diff = mu - m0
    trace_term = jnp.trace(S0_inv @ Sigma)
    return 0.5 * (trace_term + diff.T @ S0_inv @ diff - d + logdet_S0 - logdet_S)

@jit
def compute_elbo(data: jnp.ndarray, params: MoGParams, prior_mus, prior_Sigmas) -> jnp.ndarray:
    pis, mus, Lambdas = params
    log_mix = jnp.log(pis + 1e-12)[None, :] + vectorized_logpdf_prec(data, mus, Lambdas)
    ll_data = jnp.sum(logsumexp(log_mix, axis=1))
    kl_sum = 0.0
    if (prior_mus is not None) and (prior_Sigmas is not None):
        I = jnp.eye(mus.shape[1], dtype=mus.dtype)
        Sigmas = vmap(lambda La: jnp.linalg.solve(La + 1e-9*I, I))(Lambdas)
        K = pis.shape[0]
        for c in range(K):
            kl_sum += gaussian_kl(mus[c], Sigmas[c], prior_mus[c], prior_Sigmas[c])
    return ll_data - kl_sum

# ------------------------------------------------------------------------------
#                Weights <-> natural params
# ------------------------------------------------------------------------------
@jit
def weights_to_eta(pis: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-12
    ref = pis[-1] + eps
    return jnp.log(pis[:-1] + eps) - jnp.log(ref)

@jit
def eta_to_weights(eta: jnp.ndarray) -> jnp.ndarray:
    eta_full = jnp.concatenate([eta, jnp.array([0.0])])
    return jax.nn.softmax(eta_full)

# ------------------------------------------------------------------------------
#                   Log mixture helpers
# ------------------------------------------------------------------------------
@jit
def _log_mog(z: jnp.ndarray, pis: jnp.ndarray, mus: jnp.ndarray, Lambdas: jnp.ndarray) -> jnp.ndarray:
    log_comps = jnp.log(pis + 1e-12) + vmap(logpdf_gaussian_prec, (None, 0, 0))(z, mus, Lambdas)
    return logsumexp(log_comps)

@jit
def _log_prior_mog(z: jnp.ndarray, pi0: jnp.ndarray, prior_mus: jnp.ndarray, prior_Sigmas: jnp.ndarray) -> jnp.ndarray:
    def logpdf_cov(x, mu, S):
        d = x.shape[-1]
        L = jsp.linalg.cholesky(S + 1e-9*jnp.eye(d, dtype=S.dtype), lower=True)
        diff = x - mu
        y = jsp.linalg.solve_triangular(L, diff, lower=True)
        quad = y @ y
        logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diag(L) + 1e-12))
        return -0.5 * (quad + logdet_S + d * jnp.log(2.0*jnp.pi))
    log_comps = jnp.log(pi0 + 1e-12) + vmap(logpdf_cov, (None, 0, 0))(z, prior_mus, prior_Sigmas)
    return logsumexp(log_comps)

# ------------------------------------------------------------------------------
#            Anchor utilities (δ and h)
# ------------------------------------------------------------------------------
@jit
def _delta_at_z0(z0: jnp.ndarray, pis: jnp.ndarray, mus: jnp.ndarray, Lambdas: jnp.ndarray) -> jnp.ndarray:
    logN = vmap(lambda m, La: logpdf_gaussian_prec(z0, m, La))(mus, Lambdas)
    log_den = logsumexp(jnp.log(pis + 1e-12) + logN)
    return jnp.exp(logN - log_den)

@jit
def _batch_loglik_anchor(batch: jnp.ndarray, z0: jnp.ndarray, data_precision: jnp.ndarray = None) -> jnp.ndarray:
    if (batch is None) or (data_precision is None):
        return 0.0
    loglikes = vmap(lambda x: logpdf_gaussian_prec(x, z0, data_precision))(batch)
    return jnp.mean(loglikes)

@jit
def _h_strict(z0: jnp.ndarray,
              batch: jnp.ndarray,
              pis: jnp.ndarray, mus: jnp.ndarray, Lambdas: jnp.ndarray,
              prior_mus, prior_Sigmas,
              pi0: jnp.ndarray = None,
              data_precision: jnp.ndarray = None) -> jnp.ndarray:
    K = pis.shape[0]
    if pi0 is None:
        pi0 = jnp.full((K,), 1.0 / K, dtype=pis.dtype)
    log_q = _log_mog(z0, pis, mus, Lambdas)
    log_p = 0.0 if (prior_mus is None or prior_Sigmas is None) else _log_prior_mog(z0, pi0, prior_mus, prior_Sigmas)
    avg_loglike = _batch_loglik_anchor(batch, z0, data_precision)
    return (log_q - log_p) - avg_loglike

# ------------------------------------------------------------------------------
#              Stable precision from covariance
# ------------------------------------------------------------------------------
@jit
def _precision_from_cov(S: jnp.ndarray) -> jnp.ndarray:
    d = S.shape[-1]
    I = jnp.eye(d, dtype=S.dtype)
    L0 = jsp.linalg.cholesky(S + 1e-9 * I, lower=True)
    Linv0 = jsp.linalg.solve_triangular(L0, I, lower=True)
    return jsp.linalg.solve_triangular(L0.T, Linv0, lower=False)

# ------------------------------------------------------------------------------
#                  Sampling from N(mu, Λ^{-1})
# ------------------------------------------------------------------------------
@jit
def _sample_from_precision(mu: jnp.ndarray, Lambda: jnp.ndarray, key) -> jnp.ndarray:
    d = mu.shape[-1]
    L = jsp.linalg.cholesky(Lambda + 1e-9*jnp.eye(d, dtype=Lambda.dtype), lower=True)
    eps = jax.random.normal(key, (d,))
    y = jsp.linalg.solve_triangular(L.T, eps, lower=False)
    return mu + y

# ------------------------------------------------------------------------------
#                NEW: Structural A-mask helpers (SPD-safe)
# ------------------------------------------------------------------------------
def _prepare_A_mask_global_or_cluster(K: int, d: int, sparse_A, cluster_A):
    """
    Returns A_mask_K with shape (K,d,d) or None. Ensures symmetry and diag=1.
    """
    if (sparse_A is None) and (cluster_A is None):
        return None
    if cluster_A is not None:
        A = jnp.array(cluster_A, dtype=jnp.float32)
        assert A.shape == (K, d, d)
    else:
        A0 = jnp.array(sparse_A, dtype=jnp.float32)
        assert A0.shape == (d, d)
        A = jnp.tile(A0[None, :, :], (K, 1, 1))
    # symmetrize & binarize & force diag 1
    A = 0.5 * (A + jnp.swapaxes(A, -1, -2))
    A = (A > 0.5).astype(jnp.float32)
    eye = jnp.eye(d, dtype=jnp.float32)
    A = A.at[:, jnp.arange(d), jnp.arange(d)].set(1.0)  # diag must be 1
    return A

@jit
def _apply_precision_mask_spd(La: jnp.ndarray, A: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """
    Project a candidate precision La onto the masked SPD cone by:
      1) symmetrize, 2) zero disallowed off-diagonals, 3) enforce strict diagonal dominance.
    Ensures SPD without eigen-decompositions.
    """
    d = La.shape[-1]
    I = jnp.eye(d, dtype=La.dtype)
    La_sym = 0.5 * (La + La.T)
    off_mask = A * (1.0 - I)             # zeros on diag, 1 where off-diagonal is allowed
    off = La_sym * off_mask              # keep only allowed off-diagonals
    row_abs = jnp.sum(jnp.abs(off), axis=1)
    diag_new = jnp.maximum(jnp.diag(La_sym), row_abs + eps)
    return off + jnp.diag(diag_new)

@jit
def _apply_covariance_mask_from_precision(La: jnp.ndarray, A: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """
    Enforce zeros in covariance Σ and map back to precision:
      Σ* = La^{-1} -> zero disallowed off-diagonals -> enforce SPD via diagonal dominance -> Λ = (Σ*)^{-1}
    """
    d = La.shape[-1]
    I = jnp.eye(d, dtype=La.dtype)
    # Σ = Λ^{-1}
    Sigma = jnp.linalg.solve(La + 1e-9*I, I)
    Sigma_sym = 0.5 * (Sigma + Sigma.T)
    off_mask = A * (1.0 - I)
    off = Sigma_sym * off_mask
    row_abs = jnp.sum(jnp.abs(off), axis=1)
    diag_new = jnp.maximum(jnp.diag(Sigma_sym), row_abs + eps)
    Sigma_mask = off + jnp.diag(diag_new)
    # Back to precision
    L = jsp.linalg.cholesky(Sigma_mask + 1e-9*I, lower=True)
    Linv = jsp.linalg.solve_triangular(L, I, lower=True)
    La_back = jsp.linalg.solve_triangular(L.T, Linv, lower=False)
    return La_back

# Vectorized over K
@jit
def _mask_all_components(Lambdas: jnp.ndarray, A_mask_K: jnp.ndarray, use_cov_mask: bool, eps: float = 1e-6):
    def mask_one(La, A):
        return lax.cond(use_cov_mask,
                        lambda _ : _apply_covariance_mask_from_precision(La, A, eps),
                        lambda _ : _apply_precision_mask_spd(La, A, eps),
                        operand=None)
    return vmap(mask_one)(Lambdas, A_mask_K)

# ------------------------------------------------------------------------------
#            NGD update with optional A-mask projection
# ------------------------------------------------------------------------------



@partial(jit, static_argnames=('num_samples','use_h_for_weights','use_cov_mask','has_mask'))
def ngd_update_multi(params,
                     batch,
                     prior_mus,
                     prior_Sigmas,
                     beta,
                     rng_key,
                     data_precision,
                     prior_weights=None,
                     A_mask_K=None,
                     num_samples: int = 100,
                     use_h_for_weights: bool = True,
                     use_cov_mask: bool = False,
                     has_mask: bool = False,
                     spd_eps: float = 1e-6):
    pis, mus, Lambdas = params
    K, d = mus.shape
    use_prior = (prior_mus is not None) and (prior_Sigmas is not None)

    # ---- sample anchors (same as before) ----
    rng_key, k_idx, k_eps = jax.random.split(rng_key, 3)
    i_samp = jax.random.categorical(k_idx, jnp.log(pis + 1e-12), shape=(num_samples,))
    keys = jax.random.split(k_eps, num_samples)

    def _sample_from_precision(mu, La, key):
        L = jsp.linalg.cholesky(La + 1e-9*jnp.eye(d, dtype=La.dtype), lower=True)
        eps = jax.random.normal(key, (d,))
        y = jsp.linalg.solve_triangular(L.T, eps, lower=False)
        return mu + y

    z0s = vmap(lambda i, k: _sample_from_precision(mus[i], Lambdas[i], k))(i_samp, keys)

    # δ̄ and h
    def logpdf_gaussian_prec(x, mu, Lambda):
        diff = x - mu
        quad = diff @ (Lambda @ diff)
        sign, logdet_L = jnp.linalg.slogdet(Lambda)
        logdet_L = jnp.where(sign > 0, logdet_L,
                             jnp.linalg.slogdet(Lambda + 1e-9*jnp.eye(d, dtype=Lambda.dtype))[1])
        return -0.5 * (quad - logdet_L + d * jnp.log(2.0 * jnp.pi))

    def _delta_at_z0(z0):
        logN = vmap(lambda m, La: logpdf_gaussian_prec(z0, m, La))(mus, Lambdas)
        log_den = jax.scipy.special.logsumexp(jnp.log(pis + 1e-12) + logN)
        return jnp.exp(logN - log_den)

    deltas_MK = vmap(_delta_at_z0)(z0s)
    delta_bar = jnp.mean(deltas_MK, axis=0)

    def _log_mog(z):
        log_comps = jnp.log(pis + 1e-12) + vmap(logpdf_gaussian_prec, (None, 0, 0))(z, mus, Lambdas)
        return jax.scipy.special.logsumexp(log_comps)

    def _log_prior_mog(z, pi0, pmu, pSig):
        def logpdf_cov(x, mu, S):
            L = jsp.linalg.cholesky(S + 1e-9*jnp.eye(d, dtype=S.dtype), lower=True)
            diff = x - mu
            y = jsp.linalg.solve_triangular(L, diff, lower=True)
            quad = y @ y
            logdet_S = 2.0 * jnp.sum(jnp.log(jnp.diag(L) + 1e-12))
            return -0.5 * (quad + logdet_S + d * jnp.log(2.0*jnp.pi))
        log_comps = jnp.log(pi0 + 1e-12) + vmap(logpdf_cov, (None, 0, 0))(z, pmu, pSig)
        return jax.scipy.special.logsumexp(log_comps)

    def _batch_loglik_anchor(batch, z0, data_precision):
        if (batch is None) or (data_precision is None):
            return 0.0
        vals = vmap(lambda x: logpdf_gaussian_prec(x, z0, data_precision))(batch)
        return jnp.mean(vals)

    def _h(z0):
        pi0 = jnp.full((K,), 1.0 / K, dtype=pis.dtype)
        log_q = _log_mog(z0)
        log_p = 0.0 if (prior_mus is None or prior_Sigmas is None) else _log_prior_mog(z0, pi0, prior_mus, prior_Sigmas)
        avg_ll = _batch_loglik_anchor(batch, z0, data_precision)
        return (log_q - log_p) - avg_ll

    h_vals = vmap(_h)(z0s)
    h_mean = jnp.mean(h_vals)

    # Data terms
    H_data = data_precision
    barD   = jnp.mean(batch, axis=0)
    z0_bar = jnp.mean(z0s, axis=0)
    g_data = data_precision @ (z0_bar - barD)

    # Raw precision update
    if use_prior:
        def _precision_from_cov(S):
            I = jnp.eye(d, dtype=S.dtype)
            L0 = jsp.linalg.cholesky(S + 1e-9 * I, lower=True)
            Linv0 = jsp.linalg.solve_triangular(L0, I, lower=True)
            return jsp.linalg.solve_triangular(L0.T, Linv0, lower=False)
        Lambda0_all = vmap(_precision_from_cov)(prior_Sigmas)
        mu0_all     = prior_mus
    else:
        Lambda0_all = jnp.zeros_like(Lambdas)
        mu0_all     = mus

    new_Lambdas_raw = (1.0 - beta * delta_bar[:, None, None]) * Lambdas \
                      + beta * delta_bar[:, None, None] * (H_data[None, :, :] + Lambda0_all)

    # ---- APPLY MASK ONLY WHEN has_mask=True ----
    if has_mask:
        new_Lambdas = _mask_all_components(new_Lambdas_raw, A_mask_K, use_cov_mask, spd_eps)
    else:
        new_Lambdas = new_Lambdas_raw

    # Mean update
    def mean_update(mu_j, La_new_j, delta_j, Lambda0_j, mu0_j):
        step = jsp.linalg.solve(La_new_j + 1e-9*jnp.eye(d, dtype=La_new_j.dtype),
                                g_data + (Lambda0_j @ (mu_j - mu0_j)),
                                assume_a='pos')
        return mu_j - beta * delta_j * step

    new_mus = vmap(mean_update)(mus, new_Lambdas, delta_bar, Lambda0_all, mu0_all)

    # Weight update
    onehots = jax.nn.one_hot(i_samp, K, dtype=pis.dtype)
    eta_mat = onehots[:, :-1] - onehots[:, -1:]
    def weights_to_eta(p):
        eps = 1e-12
        ref = p[-1] + eps
        return jnp.log(p[:-1] + eps) - jnp.log(ref)
    def eta_to_weights(eta):
        return jax.nn.softmax(jnp.concatenate([eta, jnp.array([0.0])]))
    rho = weights_to_eta(pis)
    delta_eta = (-beta * jnp.mean(eta_mat * h_vals[:, None], axis=0)) if use_h_for_weights \
                else (-beta * jnp.mean(eta_mat, axis=0))
    rho_new = rho + delta_eta
    pis_new = eta_to_weights(rho_new)

    return MoGParams(pis_new, new_mus, new_Lambdas), rng_key, h_mean

# ------------------------------------------------------------------------------
#                    Bayes factor utilities (unchanged)
# ------------------------------------------------------------------------------
def normal_pdf(x, mean, var):
    return norm.pdf(x, loc=mean, scale=jnp.sqrt(var))

def compute_bf_for_cluster(j, X, m, v, delta):
    eps = 1e-10
    N, p = X.shape
    delta_j = delta[:, j]
    like_cluster = norm.pdf(X, loc=m[j, :], scale=jnp.sqrt(v[j, :]) + eps)
    null_mean = jnp.mean(X, axis=0)
    null_var = jnp.var(X, axis=0)
    like_null = norm.pdf(X, loc=null_mean, scale=jnp.sqrt(null_var) + eps)
    mix_like = delta_j[:, None] * like_cluster + (1 - delta_j)[:, None] * like_null
    ratio = mix_like / (like_null + eps)
    log_ratio = jnp.log(ratio + eps)
    sum_log_ratio = jnp.sum(log_ratio, axis=0)
    sum_log_ratio = jnp.clip(sum_log_ratio, a_min=-100.0, a_max=100.0)
    bf = jnp.exp(sum_log_ratio)
    return bf

def compute_bf_matrix(X, m, v, delta):
    K = m.shape[0]
    BF_matrix = jax.vmap(lambda j: compute_bf_for_cluster(j, X, m, v, delta))(jnp.arange(K))
    return BF_matrix

def compute_feature_scores_from_bf(BF_matrix, theta):
    eps = 1e-10
    BF_matrix_clipped = jnp.clip(BF_matrix, a_min=eps, a_max=1e12)
    weighted_log = jnp.sum(theta[:, None] * jnp.log(BF_matrix_clipped + eps), axis=0)
    weighted_log = jnp.clip(weighted_log, a_min=-100.0, a_max=100.0)
    score = jnp.exp(weighted_log)
    score = jnp.nan_to_num(score, nan=1.0, posinf=1e12, neginf=eps)
    return score

def compute_bayes_factors_and_scores(X, m, v, delta, theta):
    BF_matrix = compute_bf_matrix(X, m, v, delta)
    feature_scores = compute_feature_scores_from_bf(BF_matrix, theta)
    return BF_matrix, feature_scores
