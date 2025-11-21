

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
from .utility import *
# ------------------------------------------------------------------------------
#                             PRAXIS_BGM class
# ------------------------------------------------------------------------------
class Praxis_BGM:
    def __init__(
        self,
        rng_key,
        K: int,
        prior_mus: np.ndarray = None,
        prior_Sigmas: np.ndarray = None,
        init_mus: np.ndarray = None,
        init_covs: np.ndarray = None,
        prior_pis: np.ndarray = None,
        beta: float = 0.001,
        tol: float = 1e-4,
        max_iters: int = 1000,
        verbose: bool = True,
        sparse_A: np.ndarray = None,       # (d,d) global mask (0=off, 1=free)
        cluster_A: np.ndarray = None,      # (K,d,d) per-cluster mask
        prior_mus_variance: float = 10.0,
        num_samples: int = 100,
        enforce_mask: bool = False,        # <---- NEW: turn A-mask on
        mask_space: str = "precision",     # "precision" (default) or "covariance"
        spd_eps: float = 1e-6              # epsilon for SPD projection
    ):
        self.rng_key = rng_key
        self.K = K
        self.beta = beta
        self.tol = tol
        self.max_iters = max_iters
        self.verbose = verbose

        self.sparse_A = sparse_A
        self.cluster_A = cluster_A
        self.enforce_mask = enforce_mask
        self.mask_space = mask_space
        self.spd_eps = float(spd_eps)

        self.num_samples = num_samples
        self.data_precision_matrix = None

        self.user_prior_mus = prior_mus
        self.user_prior_Sigmas = prior_Sigmas
        self.prior_mus_variance = prior_mus_variance

        self.init_mus = init_mus
        self.init_covs = init_covs
        self.init_pis = prior_pis

        self.prior_mus = None
        self.prior_Sigmas = None
        self.use_mean_prior = False
        self.use_cov_prior  = False

        self.params = None
        self.elbo_history = []
        self.A_mask_K = None              # prepared (K,d,d) mask

    # ------------------------ data precision builder ---------------------------
    def _build_data_precision(self, X: jnp.ndarray, rho: float = 0.2,
                              target_trace_per_dim: float = 1.0,
                              eps: float = 1e-6) -> jnp.ndarray:
        mu = jnp.mean(X, axis=0)
        var = jnp.mean((X - mu)**2, axis=0)
        vbar = jnp.mean(var)
        var_sh = (1.0 - rho) * var + rho * vbar
        prec_diag = 1.0 / jnp.clip(var_sh, eps, None)
        d = X.shape[1]
        scale = (target_trace_per_dim * d) / jnp.sum(prec_diag)
        prec_diag = scale * prec_diag
        return jnp.diag(prec_diag.astype(jnp.float32))

    # ----------------------------- priors init ---------------------------------
    def _init_priors(self, data_jnp):
        N, d = data_jnp.shape
        if self.verbose:
            print("[Init Priors] Validating/constructing priors...")

        if self.prior_mus_variance is None:
            self.prior_mus_variance = 1.0
        scale = float(self.prior_mus_variance)

        if (self.user_prior_mus is not None) and (self.user_prior_Sigmas is not None):
            self.prior_mus   = jnp.array(self.user_prior_mus,   dtype=jnp.float32)
            self.prior_Sigmas = jnp.array(self.user_prior_Sigmas, dtype=jnp.float32) * scale
            self.use_mean_prior = True
            self.use_cov_prior  = True
            return

        if (self.user_prior_mus is not None) and (self.user_prior_Sigmas is None):
            self.prior_mus   = jnp.array(self.user_prior_mus, dtype=jnp.float32)
            I = jnp.eye(d, dtype=jnp.float32)
            self.prior_Sigmas = jnp.tile((scale * I)[None, :, :], (self.K, 1, 1))
            self.use_mean_prior = True
            self.use_cov_prior  = True
            return

        #if (self.user_prior_mus is None) and (self.user_prior_Sigmas is not None):
            #raise ValueError("Only covariance prior provided is not supported. Provide means as well, or neither.")

        I = jnp.eye(d, dtype=jnp.float32)
        self.prior_mus    = jnp.zeros((self.K, d), dtype=jnp.float32)
        self.prior_Sigmas = jnp.tile((scale * I)[None, :, :], (self.K, 1, 1))
        self.use_mean_prior = True
        self.use_cov_prior  = True
        if self.verbose:
            print("  -> No user priors. Using zero means + scaled identity covariances as priors.")

    # ---------------------------- mask prep ------------------------------------
    def _init_mask(self, d: int):
        self.A_mask_K = _prepare_A_mask_global_or_cluster(self.K, d, self.sparse_A, self.cluster_A)
        if (self.A_mask_K is not None) and self.verbose:
            mode = "precision" if self.mask_space == "precision" else "covariance"
            print(f"[Mask] Structural A-mask prepared ({mode} space).")

    # ---------------------------- param init -----------------------------------
    def _init_params(self, data_jnp):
        if self.verbose:
            print("[Init Params] Initializing mixture parameters...")
        N, d = data_jnp.shape

        init_pis = (jnp.array(self.init_pis, dtype=jnp.float32)
                    if (self.init_pis is not None)
                    else jnp.ones(self.K, dtype=jnp.float32) / self.K)

        user_supplied_none = (self.user_prior_mus is None) and (self.user_prior_Sigmas is None)

        if not user_supplied_none:
            init_mus = self.prior_mus
            init_Lambdas = vmap(_precision_from_cov)(self.prior_Sigmas)
            if self.verbose:
                print("  -> Params initialized from priors (μ := prior_mus, Λ := prior_Sigmas^{-1}).")
        else:
            if self.verbose:
                print("  -> No user priors: using sklearn BGM to initialize μ and Σ.")
            bgm = BayesianGaussianMixture(
                n_components=self.K, covariance_type="full",
                max_iter=300, init_params="kmeans",
                random_state=42, reg_covar=1e-6,
            )
            bgm.fit(np.asarray(data_jnp))
            init_mus = jnp.array(bgm.means_, dtype=jnp.float32)
            covs     = jnp.array(bgm.covariances_, dtype=jnp.float32)
            init_Lambdas = vmap(_precision_from_cov)(covs)

        # Apply mask at init (if requested)
        if self.enforce_mask and (self.A_mask_K is not None):
            use_cov_mask = (self.mask_space.lower() == "covariance")
            init_Lambdas = _mask_all_components(init_Lambdas, self.A_mask_K, use_cov_mask, self.spd_eps)

        self.params = MoGParams(init_pis, init_mus, init_Lambdas)

    # ---------------------------- training loop --------------------------------
    def _training_loop(self, data_jnp, num_iters, batch_size, early_stop, patience):
        N, d = data_jnp.shape
        self.data_precision_matrix = self._build_data_precision(
            data_jnp, rho=0.2, target_trace_per_dim=1.0)

        last_elbo = -jnp.inf
        no_change_counter = 0
        prev_assignments = None
        actual_iters = min(num_iters, self.max_iters)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > N:
            batch_size = N

        use_cov_mask = (self.mask_space.lower() == "covariance")

        for i in range(actual_iters):
            self.rng_key, key_epoch = split(self.rng_key)
            perm = permutation(key_epoch, N)
            num_batches = (N + batch_size - 1) // batch_size
            use_cov_mask = (self.mask_space.lower() == "covariance")
            has_mask = bool(self.enforce_mask and (self.A_mask_K is not None))

            for b in range(num_batches):
                start, end = b * batch_size, min((b + 1) * batch_size, N)
                batch = data_jnp[perm[start:end]]


            
                new_params, self.rng_key, _ = ngd_update_multi(
                    self.params,
                    batch,
                    self.prior_mus if self.use_mean_prior else None,
                    self.prior_Sigmas if self.use_cov_prior else None,
                    self.beta,
                    self.rng_key,
                    prior_weights=None,
                    data_precision=self.data_precision_matrix,
                    A_mask_K=self.A_mask_K,            # can be None; ignored when has_mask=False
                    num_samples=self.num_samples,
                    use_h_for_weights=True,
                    use_cov_mask=use_cov_mask,         # "precision" vs "covariance"
                    has_mask=has_mask,                 # <<< static gate (prevents A=None inside jit)
                    spd_eps=self.spd_eps
                )
                    
                self.params = new_params

            current_elbo = compute_elbo(data_jnp, self.params, self.prior_mus, self.prior_Sigmas)
            self.elbo_history.append(float(current_elbo))
            if jnp.isnan(current_elbo):
                if self.verbose:
                    print(f"[FIT] Epoch {i+1}: ELBO is NaN. Aborting for fallback.")
                return False, None

            improvement = current_elbo - last_elbo
            if self.verbose:
                print(f"[FIT] Epoch {i+1}/{actual_iters} | ELBO ≈ {current_elbo:.6f}, Δ ≈ {float(improvement):.6f}")

            if early_stop:
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
                        print("  [EarlyStop] Stable assignments => stopping.")
                    return True, float(current_elbo)

            if jnp.abs(improvement) < self.tol:
                if self.verbose:
                    print(f"  [FIT] Converged at epoch {i+1}, improvement≈{float(improvement):.3g}")
                return True, float(current_elbo)

            last_elbo = current_elbo

        return True, float(last_elbo)

    # ------------------------------- API ---------------------------------------
    def fit(self, data, num_iters=100, batch_size=50, early_stop=False, patience=2):
        data_jnp = jnp.array(data)
        N, d = data_jnp.shape
        if self.verbose:
            print(f"[FIT] Start: data=({N},{d}), K={self.K}")
            print("  => Setting up priors and mask...")

        self.elbo_history = []

        self._init_priors(data_jnp)
        self._init_mask(d)
        self._init_params(data_jnp)

        if self.verbose:
            msg_mean = "YES" if self.use_mean_prior else "NO"
            msg_cov  = "YES" if self.use_cov_prior else "NO"
            msg_mask = "ON " if (self.enforce_mask and (self.A_mask_K is not None)) else "OFF"
            print(f"  => Attempt #1 with mean prior? {msg_mean}, cov prior? {msg_cov}, A-mask {msg_mask}")

        success, final_elbo = self._training_loop(data_jnp, num_iters, batch_size, early_stop, patience)
        if success:
            if self.verbose:
                print(f"[FIT] Attempt #1 succeeded. Final ELBO ≈ {final_elbo:.4f}")
            self._print_final_summary()
            return

        if self.verbose:
            print("[FIT] Attempt #1 failed with NaN. Fallback #1 => Force covariance prior = Identity.\n")
        self.user_prior_Sigmas = np.eye(d)[None, :, :].repeat(self.K, axis=0)
        self._init_priors(data_jnp)
        self._init_params(data_jnp)
        success, final_elbo = self._training_loop(data_jnp, num_iters, batch_size, early_stop, patience)
        if success:
            if self.verbose:
                print(f"[FIT] Attempt #2 succeeded. Final ELBO ≈ {final_elbo:.4f}")
            self._print_final_summary()
            return

        if self.verbose:
            print("[FIT] Attempt #2 failed. Fallback #2 => Remove ALL user priors.\n")
        self.user_prior_mus = None
        self.user_prior_Sigmas = None
        self._init_priors(data_jnp)
        self._init_params(data_jnp)
        success, final_elbo = self._training_loop(data_jnp, num_iters, batch_size, early_stop, patience)
        if success:
            if self.verbose:
                print(f"[FIT] Attempt #3 succeeded. Final ELBO ≈ {final_elbo:.4f}")
            self._print_final_summary()
            return

        if self.verbose:
            print("[FIT] WARNING: Could not recover from NaN after all fallbacks.")
        self._print_final_summary()

    def get_posteriors(self, data):
        data_jnp = jnp.array(data)
        posterior_mus = self.params.mus
        d = posterior_mus.shape[1]
        I = jnp.eye(d, dtype=posterior_mus.dtype)
        posterior_covs = vmap(lambda La: jnp.linalg.solve(La + 1e-9*I, I))(self.params.Lambdas)
        posterior_pis = self.params.pis
        return posterior_mus, posterior_covs, posterior_pis, responsibilities(data_jnp, self.params)

    def BF_selection(self, data, top_n=20, visual=False):
        data_jnp = jnp.array(data)
        m_post, posterior_covs, theta, delta = self.get_posteriors(data_jnp)
        v_post = jnp.array([jnp.diag(Sigma) for Sigma in posterior_covs])
        BF_matrix, feature_scores = compute_bayes_factors_and_scores(data_jnp, m_post, v_post, delta, theta)
        top_features = jnp.argsort(-feature_scores)[:top_n]

        # Jeffreys scale buckets (same as your original)
        global_scores = np.array(feature_scores)
        classification = {
            "Indeterminate": [], "Positive": [], "Strong": [], "Very strong": [], "Decisive": []
        }
        for idx, score in enumerate(global_scores):
            if score < 3.2:      classification["Indeterminate"].append(idx)
            elif score < 10:     classification["Positive"].append(idx)
            elif score < 31.6:   classification["Strong"].append(idx)
            elif score < 100:    classification["Very strong"].append(idx)
            else:                classification["Decisive"].append(idx)

        if visual:
            import matplotlib.pyplot as plt
            K = m_post.shape[0]
            fig, axes = plt.subplots(K, 1, figsize=(8, 4*K))
            if K == 1: axes = [axes]
            for j in range(K):
                axes[j].bar(np.arange(top_n), np.array(m_post[j, top_features]))
                axes[j].set_title(f'Cluster {j} Mean for Top {top_n} Features')
                axes[j].set_xlabel('Feature Index')
                axes[j].set_ylabel('Mean')
            plt.tight_layout(); plt.show()

            plt.figure(figsize=(8, 4))
            valid_scores = global_scores[np.isfinite(global_scores)]
            if valid_scores.size > 0:
                plt.hist(valid_scores, bins=30, edgecolor='black')
            else:
                plt.text(0.5, 0.5, 'No valid scores to display', transform=plt.gca().transAxes,
                         ha='center', va='center')
            plt.xlabel('Global BF Score'); plt.ylabel('Frequency'); plt.title('BF Score Distribution')
            plt.tight_layout(); plt.show()

        return BF_matrix, feature_scores, top_features, classification

    def _print_final_summary(self):
        if not self.verbose:
            return
        print("\n========== FINAL SUMMARY ==========")
        print(f"  => Mean prior used? {'YES' if self.use_mean_prior else 'NO'}")
        print(f"  => Cov prior  used? {'YES' if self.use_cov_prior  else 'NO'}")
        if self.enforce_mask and (self.A_mask_K is not None):
            print(f"  => A-mask enforced in {'COVARIANCE' if self.mask_space=='covariance' else 'PRECISION'} space")
        else:
            print("  => A-mask: OFF")
        print("===================================\n")

    def predict(self, data):
        data_jnp = jnp.array(data)
        gamma = responsibilities(data_jnp, self.params)
        assignments = jnp.argmax(gamma, axis=1)
        return np.array(assignments), np.array(self.params.pis)

    def get_params(self):
        return self.params