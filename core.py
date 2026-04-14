"""High-level model API for the packaged Praxis implementation.

The :class:`Praxis_BGM` class wraps the damped global-z-prior algorithm from
``Praxis_BGM_global_z_prior_damped.py`` in a reusable library layout. The class
owns input validation, prior construction, parameter initialization, training,
and user-facing inspection helpers such as posterior extraction and feature
selection.
"""

import jax
import jax.numpy as jnp
from jax import vmap
from jax.random import choice, split
from sklearn.covariance import EmpiricalCovariance, OAS
from sklearn.mixture import BayesianGaussianMixture
import numpy as np

from .utility import *


class Praxis_BGM:
    """Prior-augmented Bayesian Gaussian mixture model with damped global-z updates.

    Parameters
    ----------
    rng_key:
        JAX PRNG key used for anchor sampling and Monte Carlo ELBO evaluation.
    K:
        Number of mixture components. Must be an integer greater than or equal
        to 2.
    prior_mus, prior_Sigmas, prior_weights:
        Optional prior mixture parameters. When omitted, the model derives
        reasonable defaults from the data using BGM and empirical covariance
        estimates.
    init_mus, init_covs, init_pis:
        Optional initialization overrides for the variational mixture.
    beta:
        Base step size used by the mean and weight updates.
    tol:
        Convergence tolerance on the Monte Carlo ELBO improvement.
    max_iters:
        Hard cap on the number of update iterations.
    verbose:
        Whether to print progress messages and training summaries.
    sparse_A, cluster_A:
        Optional global or per-cluster binary masks used to encode structural
        constraints in covariance space.
    freeze_A_zeros:
        If ``True``, covariance entries masked to zero are explicitly kept at
        zero throughout training.
    prior_mus_variance:
        Diagonal inflation factor applied when constructing prior covariances
        from user-provided prior means.
    num_samples:
        Number of latent anchor samples drawn in each Monte Carlo update.
    data_precision_int:
        Optional scalar precision for the observation model. If omitted, a
        diagonal precision is estimated from the empirical variance of the data.
    likelihood_temp:
        Global scaling factor for the observation likelihood term.
    rho_prec:
        Damping coefficient for the precision update. Smaller values produce
        more conservative covariance updates.
    rho_mu:
        Damping coefficient for the mean update.
    elbo_eval_freq:
        Frequency, in optimization iterations, at which Monte Carlo ELBO is
        evaluated and stored.
    """

    def __init__(
        self,
        rng_key,
        K: int,
        prior_mus: np.ndarray = None,
        prior_Sigmas: np.ndarray = None,
        prior_weights: np.ndarray = None,
        init_mus: np.ndarray = None,
        init_covs: np.ndarray = None,
        init_pis: np.ndarray = None,
        beta: float = 0.001,
        tol: float = 1e-4,
        max_iters: int = 1000,
        verbose: bool = True,
        sparse_A: np.ndarray = None,
        cluster_A: np.ndarray = None,
        freeze_A_zeros: bool = False,
        prior_mus_variance: float = 1.0,
        num_samples: int = 100,
        data_precision_int: int | None = None,
        likelihood_temp: float = 1.0,
        rho_prec: float = 0.05,
        rho_mu: float = 1.0,
        elbo_eval_freq: int = 10,
    ):
        if int(K) != K or K < 2:
            raise ValueError(f"K must be an integer >= 2, got {K}.")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}.")
        if tol < 0:
            raise ValueError(f"tol must be nonnegative, got {tol}.")
        if max_iters < 1:
            raise ValueError(f"max_iters must be at least 1, got {max_iters}.")
        if elbo_eval_freq < 1:
            raise ValueError(f"elbo_eval_freq must be at least 1, got {elbo_eval_freq}.")
        if (data_precision_int is not None) and (data_precision_int <= 0):
            raise ValueError(
                f"data_precision_int must be positive when provided, got {data_precision_int}."
            )
        if int(num_samples) != num_samples or num_samples < 1:
            raise ValueError(f"num_samples must be an integer >= 1, got {num_samples}.")

        self.rng_key = rng_key
        self.K = int(K)
        self.beta = beta
        self.tol = tol
        self.max_iters = max_iters
        self.verbose = verbose
        self.sparse_A = sparse_A
        self.cluster_A = cluster_A
        self.freeze_A_zeros = bool(freeze_A_zeros)
        self.num_samples = int(num_samples)
        self.data_precision_int = data_precision_int
        self.elbo_eval_freq = int(elbo_eval_freq)
        self.obs_precision_matrix = None

        self.user_prior_mus = prior_mus
        self.user_prior_Sigmas = prior_Sigmas
        self.user_prior_weights = prior_weights
        self.prior_mus_variance = prior_mus_variance

        self.init_mus = init_mus
        self.init_covs = init_covs
        self.init_pis = init_pis

        self.prior_mus = None
        self.prior_Sigmas = None
        self.prior_weights = None
        self.prior_Ls = None
        self.use_mean_prior = False
        self.use_cov_prior = False
        self.active_user_prior_mus = None
        self.active_user_prior_Sigmas = None
        self.active_user_prior_weights = None
        self.A_zero_mask = None

        self.likelihood_temp = float(likelihood_temp)
        self.rho_prec = float(rho_prec)
        self.rho_mu = float(rho_mu)

        self.params = None
        self.elbo_history = []

    def get_model_summary(self):
        """Return a compact dictionary describing the configured model state."""
        fitted = self.params is not None
        summary = {
            "variant": "Praxis_BGM_global_z_prior_damped",
            "family": "Gaussian mixture model with a global mixture prior on latent z",
            "latent_prior": "p(z) = sum_k pi0_k N(z | prior_mu_k, prior_Sigma_k)",
            "variational_family": "q(z) = sum_k pi_k N(z | mu_k, Sigma_k)",
            "observation_model": "x | z ~ N(z, Lambda_x^{-1})",
            "objective": "h(z) = log q(z) - log p(z) - scaled minibatch log-likelihood",
            "updates": {
                "anchor_sampling": "Sample anchor z from the current mixture",
                "responsibility_coupling": "All components receive updates through delta_j(z)",
                "mean_update": "Damped natural-gradient step using updated covariance",
                "covariance_update": "Damped precision blending toward Hessian targets",
                "weight_update": "Softmax-logit update using delta-weighted h(z)",
            },
            "hyperparameters": {
                "K": self.K,
                "beta": self.beta,
                "num_samples": self.num_samples,
                "likelihood_temp": self.likelihood_temp,
                "rho_prec": self.rho_prec,
                "rho_mu": self.rho_mu,
                "data_precision_int": self.data_precision_int,
                "elbo_eval_freq": self.elbo_eval_freq,
                "freeze_A_zeros": self.freeze_A_zeros,
            },
            "priors": {
                "mean_prior_source": "user" if self.active_user_prior_mus is not None else "automatic",
                "cov_prior_source": "user" if self.active_user_prior_Sigmas is not None else "automatic",
                "weight_prior_source": "user" if self.active_user_prior_weights is not None else "uniform",
            },
            "fitted": fitted,
        }

        if fitted:
            posterior_covs = jax.vmap(lambda L: L @ L.T)(self.params.Ls)
            summary["learned_state"] = {
                "pis": np.asarray(self.params.pis),
                "mus_shape": tuple(self.params.mus.shape),
                "covariances_shape": tuple(posterior_covs.shape),
                "last_mc_elbo": self.elbo_history[-1] if self.elbo_history else None,
            }

        return summary

    def _build_obs_precision(self, X: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
        """Construct the observation precision used by ``p(x | z)``.

        If ``data_precision_int`` is provided the precision is isotropic,
        otherwise the method uses inverse empirical variance per feature.
        """
        d = X.shape[1]
        if self.data_precision_int is not None:
            return float(self.data_precision_int) * jnp.eye(d, dtype=jnp.float32)

        var = jnp.var(X, axis=0)
        prec_diag = 1.0 / jnp.clip(var, min=eps)
        return jnp.diag(prec_diag.astype(jnp.float32))

    def _init_priors(
        self,
        data_jnp,
        user_prior_mus=_DEFAULT,
        user_prior_Sigmas=_DEFAULT,
        user_prior_weights=_DEFAULT,
    ):
        """Validate or construct the prior mixture used by the algorithm.

        This method centralizes all prior fallback behavior so that training and
        test code can share the same initialization logic.
        """
        _, d = data_jnp.shape
        if self.verbose:
            print("[Init Priors] Checking user-provided priors...")

        if user_prior_mus is _DEFAULT:
            user_prior_mus = self.user_prior_mus
        if user_prior_Sigmas is _DEFAULT:
            user_prior_Sigmas = self.user_prior_Sigmas
        if user_prior_weights is _DEFAULT:
            user_prior_weights = self.user_prior_weights

        self.active_user_prior_mus = user_prior_mus
        self.active_user_prior_Sigmas = user_prior_Sigmas
        self.active_user_prior_weights = user_prior_weights

        if self.prior_mus_variance is None:
            self.prior_mus_variance = 1.0

        self.use_mean_prior = False
        self.use_cov_prior = False
        self.A_zero_mask = None

        if self.freeze_A_zeros:
            self.A_zero_mask = _prepare_zero_freeze_mask(
                self.K,
                d,
                sparse_A=self.sparse_A,
                cluster_A=self.cluster_A,
            )
            if self.verbose:
                if self.A_zero_mask is None:
                    print("  => freeze_A_zeros=True but no A mask was provided; nothing will be frozen.")
                else:
                    print("  => Zero-valued A entries will be frozen during training.")

        def empirical_cov(x):
            """Choose an empirical covariance estimator based on aspect ratio."""
            n, d_ = x.shape
            x_np = np.array(x)
            if d_ > 2 * n:
                if self.verbose:
                    print("     [OAS] Using Oracle Approximating Shrinkage (OAS).")
                cov = OAS().fit(x_np).covariance_.astype(np.float32)
            else:
                if self.verbose:
                    print("     [EmpiricalCovariance] Using standard empirical covariance.")
                cov = EmpiricalCovariance().fit(x_np).covariance_.astype(np.float32)
            return jnp.array(cov)

        if (user_prior_mus is not None) and (user_prior_Sigmas is not None):
            if self.verbose:
                print("  => Using user-provided prior means & covariances.")
            prior_mus_np = _validate_means("prior_mus", user_prior_mus, self.K)
            prior_Sigmas_np = _validate_covariances(
                "prior_Sigmas",
                user_prior_Sigmas,
                self.K,
                d=prior_mus_np.shape[1],
            )
            self.prior_mus = jnp.array(prior_mus_np, dtype=jnp.float32)
            self.prior_Sigmas = jnp.array(prior_Sigmas_np, dtype=jnp.float32)
            diag = jnp.einsum("kii->ki", self.prior_Sigmas)
            new_diag = self.prior_mus_variance * diag
            self.prior_Sigmas = self.prior_Sigmas.at[:, jnp.arange(d), jnp.arange(d)].set(new_diag)
            self.use_mean_prior = True
            self.use_cov_prior = True
        elif (user_prior_mus is not None) and (user_prior_Sigmas is None):
            if self.verbose:
                print("  => Only prior means provided.")
            prior_mus_np = _validate_means("prior_mus", user_prior_mus, self.K)
            self.prior_mus = jnp.array(prior_mus_np, dtype=jnp.float32)
            if (self.sparse_A is not None) or (self.cluster_A is not None):
                emp_cov_mat = empirical_cov(data_jnp)
                self.prior_Sigmas = jnp.tile(emp_cov_mat[None, :, :], (self.K, 1, 1))
            else:
                self.prior_Sigmas = jnp.tile(
                    self.prior_mus_variance * jnp.eye(d, dtype=jnp.float32)[None, :, :],
                    (self.K, 1, 1),
                )
            self.use_mean_prior = True
            self.use_cov_prior = True
        elif (user_prior_mus is None) and (user_prior_Sigmas is not None):
            if self.verbose:
                print("  => Only prior covariances provided; using BGM for prior means.")
            bgm = BayesianGaussianMixture(
                n_components=self.K,
                covariance_type="tied",
                max_iter=300,
                random_state=42,
            )
            bgm.fit(np.array(data_jnp))
            self.prior_mus = jnp.array(bgm.means_.astype(np.float32), dtype=jnp.float32)
            prior_Sigmas_np = _validate_covariances("prior_Sigmas", user_prior_Sigmas, self.K, d=d)
            self.prior_Sigmas = jnp.array(prior_Sigmas_np, dtype=jnp.float32)
            self.use_mean_prior = True
            self.use_cov_prior = True
        else:
            if (self.sparse_A is not None) or (self.cluster_A is not None):
                if self.verbose:
                    print("  => No user priors, using BGM(full) means + empirical cov.")
                bgm = BayesianGaussianMixture(
                    n_components=self.K,
                    covariance_type="full",
                    max_iter=300,
                    random_state=42,
                )
                bgm.fit(np.array(data_jnp))
                self.prior_mus = jnp.array(bgm.means_.astype(np.float32), dtype=jnp.float32)
                emp_cov_mat = empirical_cov(data_jnp)
                self.prior_Sigmas = jnp.tile(emp_cov_mat[None, :, :], (self.K, 1, 1))
            else:
                if self.verbose:
                    print("  => No user priors, using BGM(full) means + identity cov.")
                bgm = BayesianGaussianMixture(
                    n_components=self.K,
                    covariance_type="full",
                    max_iter=300,
                    random_state=42,
                )
                bgm.fit(np.array(data_jnp))
                self.prior_mus = jnp.array(bgm.means_.astype(np.float32), dtype=jnp.float32)
                self.prior_Sigmas = jnp.tile(jnp.eye(d, dtype=jnp.float32)[None, :, :], (self.K, 1, 1))
            self.use_mean_prior = True
            self.use_cov_prior = True

        if self.prior_mus.shape != (self.K, d):
            raise ValueError(f"prior_mus must have shape ({self.K}, {d}), got {self.prior_mus.shape}.")
        if self.prior_Sigmas.shape != (self.K, d, d):
            raise ValueError(
                f"prior_Sigmas must have shape ({self.K}, {d}, {d}), got {self.prior_Sigmas.shape}."
            )

        if user_prior_weights is not None:
            pw = _validate_prob_vector("prior_weights", user_prior_weights, self.K)
            self.prior_weights = jnp.array(pw, dtype=jnp.float32)
        else:
            self.prior_weights = jnp.ones((self.K,), dtype=jnp.float32) / self.K

        if (self.sparse_A is not None) and (self.cluster_A is not None):
            if self.verbose:
                print("  => Applying overall + cluster masks to prior covariances.")
            overall_A = jnp.array(self.sparse_A, dtype=jnp.float32)
            cluster_A_jnp = jnp.array(self.cluster_A, dtype=jnp.float32)
            if overall_A.shape != (d, d):
                raise ValueError(f"sparse_A must have shape ({d}, {d}), got {overall_A.shape}.")
            if cluster_A_jnp.shape != (self.K, d, d):
                raise ValueError(
                    f"cluster_A must have shape ({self.K}, {d}, {d}), got {cluster_A_jnp.shape}."
                )
            self.prior_Sigmas = jnp.stack(
                [self.prior_Sigmas[i] * overall_A * cluster_A_jnp[i] for i in range(self.K)],
                axis=0,
            )
        elif self.cluster_A is not None:
            if self.verbose:
                print("  => Applying cluster-specific mask to prior covariances.")
            cluster_A_jnp = jnp.array(self.cluster_A, dtype=jnp.float32)
            if cluster_A_jnp.shape != (self.K, d, d):
                raise ValueError(
                    f"cluster_A must have shape ({self.K}, {d}, {d}), got {cluster_A_jnp.shape}."
                )
            self.prior_Sigmas = jnp.stack(
                [self.prior_Sigmas[i] * cluster_A_jnp[i] for i in range(self.K)],
                axis=0,
            )
        elif self.sparse_A is not None:
            if self.verbose:
                print("  => Applying overall mask to prior covariances.")
            overall_A = jnp.array(self.sparse_A, dtype=jnp.float32)
            if overall_A.shape != (d, d):
                raise ValueError(f"sparse_A must have shape ({d}, {d}), got {overall_A.shape}.")
            self.prior_Sigmas = jnp.stack([self.prior_Sigmas[i] * overall_A for i in range(self.K)], axis=0)

        self.prior_Sigmas = _stabilize_covariances(self.prior_Sigmas)
        self.prior_Ls = _prior_Ls_from_covs(self.prior_Sigmas)

    def _init_params(self, data_jnp):
        """Initialize variational mixture weights, means, and covariance factors."""
        if self.verbose:
            print("[Init Params] Initializing mixture parameters...")
        _, d = data_jnp.shape

        if self.init_pis is not None:
            init_pis = jnp.array(_validate_prob_vector("init_pis", self.init_pis, self.K), dtype=jnp.float32)
        else:
            init_pis = jnp.ones(self.K, dtype=jnp.float32) / self.K

        if self.init_mus is not None:
            init_mus = jnp.array(_validate_means("init_mus", self.init_mus, self.K), dtype=jnp.float32)
        elif self.use_mean_prior:
            init_mus = self.prior_mus
        else:
            init_mus = jnp.zeros((self.K, d), dtype=jnp.float32)

        if init_mus.shape != (self.K, d):
            raise ValueError(f"Initial means must have shape ({self.K}, {d}), got {init_mus.shape}.")

        if self.init_covs is not None:
            init_covs_np = _validate_covariances("init_covs", self.init_covs, self.K, d=d)
            stable_covs = _stabilize_covariances(jnp.array(init_covs_np, dtype=jnp.float32))
            init_Ls = _prior_Ls_from_covs(stable_covs)
        elif self.use_cov_prior:
            init_Ls = self.prior_Ls
        else:
            init_Ls = jnp.tile(jnp.eye(d, dtype=jnp.float32)[None, :, :], (self.K, 1, 1))

        if self.freeze_A_zeros and (self.A_zero_mask is not None):
            init_covs = vmap(lambda L: L @ L.T)(init_Ls)
            init_Ls = vmap(_chol_from_masked_covariance)(init_covs, self.A_zero_mask)

        self.params = MoGParams(init_pis, init_mus, init_Ls)

    def _training_loop(self, data_jnp, num_iters, batch_size, early_stop, patience):
        """Run the damped Monte Carlo optimization loop.

        Returns
        -------
        tuple[bool, float | None]
            A success flag and the final finite ELBO when available.
        """
        N, _ = data_jnp.shape
        self.obs_precision_matrix = self._build_obs_precision(data_jnp)
        self.N_total = N

        last_elbo = -jnp.inf
        beta_local = float(self.beta)
        no_change_counter = 0
        bad_step_counter = 0
        prev_assignments = None
        last_good_params = self.params
        actual_iters = min(num_iters, self.max_iters)
        freeze_mask_now = bool(self.freeze_A_zeros and (self.A_zero_mask is not None))

        for i in range(actual_iters):
            self.rng_key, key_batch = split(self.rng_key)
            if batch_size < 1:
                raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
            if batch_size > N:
                raise ValueError(f"batch_size = {batch_size} > N = {N}")

            indices = choice(key_batch, N, shape=(batch_size,), replace=False)
            batch = data_jnp[indices]

            candidate_params, self.rng_key, h_anchor = ngd_update(
                last_good_params,
                batch,
                self.prior_mus,
                self.prior_Sigmas,
                beta_local,
                self.rng_key,
                prior_weights=self.prior_weights,
                prior_Ls=self.prior_Ls,
                obs_precision=self.obs_precision_matrix,
                N_total=self.N_total,
                tau=self.likelihood_temp,
                num_samples=self.num_samples,
                A_zero_mask=self.A_zero_mask,
                freeze_A_zeros=freeze_mask_now,
                rho_prec=self.rho_prec,
                rho_mu=self.rho_mu,
            )
            candidate_ok = bool(_params_are_finite(candidate_params)) and bool(jnp.isfinite(h_anchor))

            should_eval_elbo = ((i + 1) % self.elbo_eval_freq == 0) or ((i + 1) == actual_iters)
            ll_data = reg = current_elbo = None
            if should_eval_elbo:
                self.rng_key, key_elbo = split(self.rng_key)
                ll_data, reg, current_elbo = compute_elbo_terms(
                    data_jnp,
                    candidate_params,
                    self.prior_mus,
                    self.prior_Sigmas,
                    self.obs_precision_matrix,
                    key_elbo,
                    self.prior_weights,
                    prior_Ls=self.prior_Ls,
                    tau=self.likelihood_temp,
                    num_samples=self.num_samples,
                )
                candidate_ok = (
                    candidate_ok
                    and bool(jnp.isfinite(ll_data))
                    and bool(jnp.isfinite(reg))
                    and bool(jnp.isfinite(current_elbo))
                )

            if not candidate_ok:
                self.params = last_good_params
                beta_local = max(beta_local * 0.5, 1e-8)
                bad_step_counter += 1
                if self.verbose:
                    pieces = [f"[FIT] Iter {i+1}/{actual_iters}: rejected non-finite update"]
                    if should_eval_elbo and (current_elbo is not None):
                        pieces.append(f"MC ELBO candidate = {float(current_elbo):.6f}")
                    pieces.append(f"retry beta = {beta_local:.3e}")
                    pieces.append(f"rejections = {bad_step_counter}")
                    print(" | ".join(pieces))
                continue

            self.params = candidate_params
            last_good_params = candidate_params
            bad_step_counter = 0
            beta_local = min(float(self.beta), beta_local * 1.05)

            if should_eval_elbo:
                self.elbo_history.append(float(current_elbo))
                improvement = current_elbo - last_elbo
                if self.verbose:
                    ll_per_obs = ll_data / N
                    reg_per_comp = reg / self.K
                    print(
                        f"[FIT] Iter {i+1}/{actual_iters} | "
                        f"MC ELBO ≈ {current_elbo:.6f} | "
                        f"Δ ≈ {improvement:.6f} | "
                        f"data term ≈ {ll_data:.6f} ({ll_per_obs:.6f}/obs) | "
                        f"reg term ≈ {reg:.6f} ({reg_per_comp:.6f}/comp) | "
                        f"avg h(z) ≈ {float(h_anchor):.6f} | "
                        f"beta ≈ {beta_local:.3e}"
                    )

                if jnp.abs(improvement) < self.tol:
                    if self.verbose:
                        print(f"  [FIT] Converged at iteration {i+1}, improvement≈{improvement:.3g}")
                    return True, float(current_elbo)

                last_elbo = current_elbo

            if early_stop and ((i + 1) % 10 == 0):
                cluster_assignments = np.argmax(responsibilities(data_jnp, self.params), axis=1)
                if (prev_assignments is not None) and np.array_equal(cluster_assignments, prev_assignments):
                    no_change_counter += 1
                    if self.verbose:
                        print(f"  [EarlyStop] No change in assignments: {no_change_counter}/{patience}")
                else:
                    no_change_counter = 0
                prev_assignments = cluster_assignments

                if no_change_counter >= patience:
                    if self.verbose:
                        print("  [EarlyStop] No change for patience limit => stopping.")
                    final_elbo = last_elbo
                    if not np.isfinite(final_elbo):
                        self.rng_key, key_elbo = split(self.rng_key)
                        ll_data, reg, final_elbo = compute_elbo_terms(
                            data_jnp,
                            self.params,
                            self.prior_mus,
                            self.prior_Sigmas,
                            self.obs_precision_matrix,
                            key_elbo,
                            self.prior_weights,
                            prior_Ls=self.prior_Ls,
                            tau=self.likelihood_temp,
                            num_samples=self.num_samples,
                        )
                        self.elbo_history.append(final_elbo)
                        if self.verbose:
                            print(
                                f"  [EarlyStop] Final MC ELBO ≈ {float(final_elbo):.6f} | "
                                f"data term ≈ {float(ll_data):.6f} | "
                                f"reg term ≈ {float(reg):.6f}"
                            )
                    return True, float(final_elbo)

        if not np.isfinite(last_elbo):
            self.rng_key, key_elbo = split(self.rng_key)
            ll_data, reg, last_elbo = compute_elbo_terms(
                data_jnp,
                last_good_params,
                self.prior_mus,
                self.prior_Sigmas,
                self.obs_precision_matrix,
                key_elbo,
                self.prior_weights,
                prior_Ls=self.prior_Ls,
                tau=self.likelihood_temp,
                num_samples=self.num_samples,
            )
            if bool(jnp.isfinite(last_elbo)):
                self.params = last_good_params
                self.elbo_history.append(float(last_elbo))
                if self.verbose:
                    print(
                        f"[FIT] Final recovered MC ELBO ≈ {float(last_elbo):.6f} | "
                        f"data term ≈ {float(ll_data):.6f} | "
                        f"reg term ≈ {float(reg):.6f}"
                    )
                return True, float(last_elbo)
            return False, None

        return True, float(last_elbo)

    def fit(self, data, num_iters=50, batch_size=50, early_stop=False, patience=2):
        """Fit the variational mixture model to a two-dimensional data matrix.

        The method validates runtime arguments, initializes priors and
        variational parameters, and applies two fallback strategies if the
        initial training attempt becomes numerically unstable.
        """
        data_jnp = jnp.array(data)
        if data_jnp.ndim != 2:
            raise ValueError(f"data must be a 2D array of shape (N, d), got {data_jnp.shape}.")
        if not np.all(np.isfinite(np.asarray(data_jnp))):
            raise ValueError("data must contain only finite values.")
        if int(num_iters) != num_iters:
            raise ValueError(f"num_iters must be an integer, got {num_iters}.")
        if int(batch_size) != batch_size:
            raise ValueError(f"batch_size must be an integer, got {batch_size}.")
        if int(patience) != patience:
            raise ValueError(f"patience must be an integer, got {patience}.")
        num_iters = int(num_iters)
        batch_size = int(batch_size)
        patience = int(patience)
        if num_iters < 1:
            raise ValueError(f"num_iters must be at least 1, got {num_iters}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
        if patience < 1:
            raise ValueError(f"patience must be at least 1, got {patience}.")
        N, d = data_jnp.shape
        if self.verbose:
            print(f"[FIT] Starting training with data shape = ({N}, {d}), K = {self.K}")
            print(f"  => Checking user priors & setting up initial state (num_samples = {self.num_samples})...")

        self.elbo_history = []

        self._init_priors(data_jnp)
        self._init_params(data_jnp)
        if self.verbose:
            print("  => Attempt #1 with current priors.")
        success, final_elbo = self._training_loop(data_jnp, num_iters, batch_size, early_stop, patience)
        if success:
            if self.verbose:
                print(f"[FIT] Attempt #1 succeeded. Final ELBO ≈ {final_elbo:.4f}")
            self._print_final_summary()
            return

        if self.verbose:
            print("[FIT] Attempt #1 failed with NaN. Fallback #1 => Force prior covariance = identity.\n")
        fallback_prior_Sigmas = np.eye(d, dtype=np.float32)[None, :, :].repeat(self.K, axis=0)
        self._init_priors(data_jnp, user_prior_Sigmas=fallback_prior_Sigmas)
        self._init_params(data_jnp)
        success, final_elbo = self._training_loop(data_jnp, num_iters, batch_size, early_stop, patience)
        if success:
            if self.verbose:
                print(f"[FIT] Attempt #2 succeeded. Final ELBO ≈ {final_elbo:.4f}")
            self._print_final_summary()
            return

        if self.verbose:
            print("[FIT] Attempt #2 failed. Fallback #2 => Removing all user priors.\n")
        self._init_priors(data_jnp, user_prior_mus=None, user_prior_Sigmas=None, user_prior_weights=None)
        self._init_params(data_jnp)
        success, final_elbo = self._training_loop(data_jnp, num_iters, batch_size, early_stop, patience)
        if success:
            if self.verbose:
                print(f"[FIT] Attempt #3 succeeded. Final ELBO ≈ {final_elbo:.4f}")
            self._print_final_summary()
            return

        if self.verbose:
            print("[FIT] WARNING: Could not recover from NaN even after all fallbacks.")
        self._print_final_summary()

    def get_posteriors(self, data):
        """Return posterior means, covariances, weights, and responsibilities."""
        data_jnp = jnp.array(data)
        posterior_mus = self.params.mus
        posterior_covs = jax.vmap(lambda L: L @ L.T)(self.params.Ls)
        posterior_pis = self.params.pis
        return posterior_mus, posterior_covs, posterior_pis, responsibilities(data_jnp, self.params)

    def BF_selection(self, data, top_n=20, visual=False):
        """Compute Bayes-factor-based feature importance scores.

        Parameters
        ----------
        data:
            Observed data matrix used to evaluate responsibilities and posterior
            feature evidence.
        top_n:
            Number of highest-scoring features to return.
        visual:
            If ``True``, also display simple summary plots for posterior means
            and the feature-score distribution.
        """
        data_jnp = jnp.array(data)
        m_post, posterior_covs, theta, delta = self.get_posteriors(data_jnp)
        v_post = jnp.array([jnp.diag(Sigma) for Sigma in posterior_covs])
        BF_matrix, feature_scores = compute_bayes_factors_and_scores(data_jnp, m_post, v_post, delta, theta)
        top_features = jnp.argsort(-feature_scores)[:top_n]

        global_scores = np.array(feature_scores)
        classification = {
            "Indeterminate": [],
            "Positive": [],
            "Strong": [],
            "Very strong": [],
            "Decisive": [],
        }
        for idx, score in enumerate(global_scores):
            if score < 3.2:
                classification["Indeterminate"].append(idx)
            elif score < 10:
                classification["Positive"].append(idx)
            elif score < 31.6:
                classification["Strong"].append(idx)
            elif score < 100:
                classification["Very strong"].append(idx)
            else:
                classification["Decisive"].append(idx)

        if visual:
            import matplotlib.pyplot as plt

            K = m_post.shape[0]
            fig, axes = plt.subplots(K, 1, figsize=(8, 4 * K))
            if K == 1:
                axes = [axes]
            for j in range(K):
                axes[j].bar(np.arange(top_n), np.array(m_post[j, top_features]))
                axes[j].set_title(f"Cluster {j} Mean for Top {top_n} Features")
                axes[j].set_xlabel("Feature Index (Top features)")
                axes[j].set_ylabel("Mean")
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(8, 4))
            valid_scores = global_scores[np.isfinite(global_scores)]
            if valid_scores.size > 0:
                plt.hist(valid_scores, bins=30, edgecolor="black")
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No valid scores to display",
                    transform=plt.gca().transAxes,
                    ha="center",
                    va="center",
                )
            plt.xlabel("Global BF Score")
            plt.ylabel("Frequency")
            plt.title("Distribution of Global BF Scores")
            plt.tight_layout()
            plt.show()

        return BF_matrix, feature_scores, top_features, classification

    def _print_final_summary(self):
        """Print a human-readable summary of the fitted model configuration."""
        if self.verbose:
            summary = self.get_model_summary()
            print("\n========== FINAL SUMMARY ==========")
            print(f"  => Variant: {summary['variant']}")
            print("  => Global prior on z implemented as a Gaussian mixture prior.")
            print("  => Observation model: x | z ~ N(z, Lambda_x^{-1}).")
            print("  => Optimization: anchor-based VON / damped natural-gradient updates.")
            if self.active_user_prior_mus is not None:
                print("  => Prior means are user-provided or partially user-informed.")
            else:
                print("  => Prior means initialized from BGM.")
            if self.active_user_prior_Sigmas is not None:
                print("  => Prior covariances are user-provided or partially user-informed.")
            else:
                print("  => Prior covariances initialized automatically.")
            if self.active_user_prior_weights is not None:
                print("  => Prior weights are user-provided.")
            else:
                print("  => Prior weights are uniform.")
            if self.freeze_A_zeros and (self.A_zero_mask is not None):
                print("  => Zero-valued A entries are frozen during training.")
            if self.elbo_history:
                print(f"  => Last MC ELBO: {float(self.elbo_history[-1]):.6f}")
            print("===================================\n")

    def predict(self, data):
        """Predict hard cluster assignments and return them with learned weights."""
        data_jnp = jnp.array(data)
        gamma = responsibilities(data_jnp, self.params)
        assignments = jnp.argmax(gamma, axis=1)
        return np.array(assignments), np.array(self.params.pis)

    def get_params(self):
        """Return the packed variational parameters for advanced downstream use."""
        return self.params
