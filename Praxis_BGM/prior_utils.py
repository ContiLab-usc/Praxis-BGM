"""
prior_utils.py

Utilities to:
1) Align source and target datasets on common features, apply QC, select
   top-N most variable features (based on source), and optionally standardize.
2) Construct Gaussian mixture priors (means, covariances, weights) from
   a labeled source dataset for Praxis-BGM.

"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------
# 1. Source–target alignment + QC + top-N most variable feature selection
# ----------------------------------------------------------------------

def _coerce_to_dataframe(
    X,
    feature_names: Optional[np.ndarray] = None,
    sample_axis: int = 0,
) -> pd.DataFrame:
    """
    Convert input to a pandas DataFrame with samples as rows, features as columns.

    Parameters
    ----------
    X : array-like or DataFrame
        Input data. If ndarray, shape should be (n_samples, n_features) or
        (n_features, n_samples) depending on `sample_axis`.
    feature_names : array-like of str, optional
        Column names to use if X is an ndarray. If None and X is ndarray,
        feature names will be 'f0', 'f1', ...
    sample_axis : int, default=0
        Axis corresponding to samples in X. If 1, X will be transposed.

    Returns
    -------
    df : DataFrame
        DataFrame with samples as rows and features as columns.
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()

    X = np.asarray(X)
    if sample_axis == 1:
        X = X.T  # (n_samples, n_features)

    n_samples, n_features = X.shape
    if feature_names is None:
        cols = [f"f{i}" for i in range(n_features)]
    else:
        cols = list(feature_names)
        if len(cols) != n_features:
            raise ValueError(
                f"feature_names length {len(cols)} does not match "
                f"n_features={n_features}"
            )

    return pd.DataFrame(X, columns=cols)


def prepare_source_target_datasets(
    source,
    target,
    source_feature_names: Optional[np.ndarray] = None,
    target_feature_names: Optional[np.ndarray] = None,
    sample_axis: int = 0,
    top_n: Optional[int] = None,
    standardize: bool = True,
    impute_strategy: str = "mean",
    min_non_null_fraction: float = 0.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Align source and target datasets on common features, perform QC,
    select top-N most variable features (based on source), and optionally
    standardize each feature using statistics from the source.

    Parameters
    ----------
    source : array-like or DataFrame
        Source data (with labels or priors). Samples x features or features x samples
        depending on `sample_axis`.
    target : array-like or DataFrame
        Target data (unlabeled). Same shape convention as `source`.
    source_feature_names : array-like of str, optional
        Feature names for source if `source` is ndarray.
    target_feature_names : array-like of str, optional
        Feature names for target if `target` is ndarray.
    sample_axis : int, default=0
        Axis for samples in both source and target (0 = rows, 1 = columns).
    top_n : int, optional
        If given, select the top-N most variable features based on source.
        If None, use all intersected features after QC.
    standardize : bool, default=True
        If True, fit a StandardScaler on the source data and apply to both
        source and target (feature-wise mean 0, std 1 based on source).
    impute_strategy : {"mean", "median", "most_frequent", "constant"}
        Strategy for SimpleImputer to handle missing values.
    min_non_null_fraction : float, default=0.5
        Minimum fraction of non-null entries required per feature (in source AND target).
        Features with too many missing values will be dropped.
    verbose : bool, default=True
        If True, print basic QC and alignment information.

    Returns
    -------
    X_source_aligned : ndarray of shape (n_source_samples, n_features_used)
    X_target_aligned : ndarray of shape (n_target_samples, n_features_used)
    feature_names_used : ndarray of str, shape (n_features_used,)
    metadata : dict
        Dictionary containing intermediate information:
          - "features_intersection"
          - "features_after_qc"
          - "variance_source"
          - "scaler" (StandardScaler or None)
          - "imputer" (SimpleImputer)
    """
    # 1) Coerce to DataFrame with same orientation
    src_df = _coerce_to_dataframe(source, source_feature_names, sample_axis)
    tgt_df = _coerce_to_dataframe(target, target_feature_names, sample_axis)

    if verbose:
        print(f"[Align] Source shape (raw): {src_df.shape}")
        print(f"[Align] Target shape (raw): {tgt_df.shape}")

    # 2) Intersect feature sets
    common_features = sorted(set(src_df.columns).intersection(set(tgt_df.columns)))
    if len(common_features) == 0:
        raise ValueError("No overlapping features between source and target.")

    src_df = src_df[common_features]
    tgt_df = tgt_df[common_features]

    if verbose:
        print(f"[Align] Common features: {len(common_features)}")

    # 3) Basic QC: drop features with too many missing values or zero variance
    #    Compute non-null fraction in BOTH source and target
    src_non_null_frac = src_df.notna().mean(axis=0)
    tgt_non_null_frac = tgt_df.notna().mean(axis=0)
    non_null_mask = (src_non_null_frac >= min_non_null_fraction) & (
        tgt_non_null_frac >= min_non_null_fraction
    )
    features_after_na = np.array(common_features)[non_null_mask.to_numpy()]

    src_df = src_df[features_after_na]
    tgt_df = tgt_df[features_after_na]

    # Zero-variance filter based on source
    src_var = src_df.var(axis=0, ddof=1)
    nonzero_var_mask = src_var > 0.0
    features_after_qc = features_after_na[nonzero_var_mask.to_numpy()]

    src_df = src_df[features_after_qc]
    tgt_df = tgt_df[features_after_qc]

    if verbose:
        print(f"[QC] Features after NA filter: {len(features_after_na)}")
        print(f"[QC] Features after zero-variance filter: {len(features_after_qc)}")

    # 4) Impute missing values using source statistics (fit on source, apply to both)
    imputer = SimpleImputer(strategy=impute_strategy)
    imputer.fit(src_df.values)

    src_imputed = imputer.transform(src_df.values)
    tgt_imputed = imputer.transform(tgt_df.values)

    # 5) Top-N most variable features (based on imputed source)
    var_source = np.var(src_imputed, axis=0, ddof=1)
    feature_names_used = np.array(features_after_qc)

    if top_n is not None and top_n < len(feature_names_used):
        idx_sorted = np.argsort(-var_source)  # descending
        idx_keep = idx_sorted[:top_n]
        src_imputed = src_imputed[:, idx_keep]
        tgt_imputed = tgt_imputed[:, idx_keep]
        feature_names_used = feature_names_used[idx_keep]
        var_source = var_source[idx_keep]

        if verbose:
            print(f"[VarSelect] Selected top {top_n} most variable features.")

    # 6) Optional standardization (fit on source, apply to both)
    scaler = None
    if standardize:
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.fit(src_imputed)
        src_scaled = scaler.transform(src_imputed)
        tgt_scaled = scaler.transform(tgt_imputed)
    else:
        src_scaled = src_imputed
        tgt_scaled = tgt_imputed

    metadata = {
        "features_intersection": np.array(common_features),
        "features_after_qc": features_after_qc,
        "variance_source": var_source,
        "scaler": scaler,
        "imputer": imputer,
    }

    if verbose:
        print(f"[Final] Source aligned shape: {src_scaled.shape}")
        print(f"[Final] Target aligned shape: {tgt_scaled.shape}")

    return src_scaled, tgt_scaled, feature_names_used, metadata


# ----------------------------------------------------------------------
# 2. Construct priors from source dataset + cluster labels
# ----------------------------------------------------------------------

def build_gaussian_priors_from_source(
    X_source: np.ndarray,
    labels_source: np.ndarray,
    K: Optional[int] = None,
    shrinkage: float = 1e-6,
    diag_only: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct Gaussian mixture priors from a labeled source dataset.

    This function computes cluster-wise:
      - prior means:  μ_k
      - prior covariances: Σ_k  (with ridge shrinkage)
      - prior weights: π_k      (empirical frequencies)

    These outputs can be passed directly to Praxis_BGM as:
      prior_mus    -> μ_k
      prior_Sigmas -> Σ_k
      prior_pis    -> π_k   (if you wish to use initial mixing weights)

    Parameters
    ----------
    X_source : ndarray of shape (n_samples, n_features)
        Aligned and optionally standardized source data.
    labels_source : array-like of shape (n_samples,)
        Cluster labels for source samples. Can be ints or strings.
    K : int, optional
        Number of clusters. If None, inferred from unique labels.
    shrinkage : float, default=1e-6
        Ridge term added to the diagonal of each covariance matrix to ensure
        positive-definiteness and numerical stability.
    diag_only : bool, default=False
        If True, construct diagonal covariance matrices using per-feature
        variances within each cluster. If False, use full empirical covariance.
    verbose : bool, default=True
        If True, print cluster counts and basic prior information.

    Returns
    -------
    prior_mus : ndarray of shape (K, d)
        Cluster-wise mean vectors.
    prior_Sigmas : ndarray of shape (K, d, d)
        Cluster-wise covariance matrices (with shrinkage).
    prior_pis : ndarray of shape (K,)
        Cluster-wise mixing weights (empirical frequencies).
    """
    X_source = np.asarray(X_source)
    labels_source = np.asarray(labels_source)

    if X_source.ndim != 2:
        raise ValueError("X_source must be 2D (n_samples, n_features).")
    n_samples, d = X_source.shape

    unique_labels, label_indices = np.unique(labels_source, return_inverse=True)
    if K is None:
        K = len(unique_labels)
    else:
        if K != len(unique_labels):
            if verbose:
                print(
                    f"[Priors] Provided K={K}, but found {len(unique_labels)} unique "
                    f"labels. Using K={len(unique_labels)} from data."
                )
            K = len(unique_labels)

    prior_mus = np.zeros((K, d), dtype=float)
    prior_Sigmas = np.zeros((K, d, d), dtype=float)
    prior_pis = np.zeros((K,), dtype=float)

    if verbose:
        print(f"[Priors] Building priors from source with K={K}, d={d}")
        for k, ulab in enumerate(unique_labels):
            ck = np.sum(label_indices == k)
            print(f"  - Cluster {k} (label={ulab}): n={ck}")

    for k in range(K):
        idx_k = np.where(label_indices == k)[0]
        if idx_k.size == 0:
            # Empty cluster label (should not usually happen)
            prior_mus[k, :] = 0.0
            prior_Sigmas[k, :, :] = np.eye(d) * shrinkage
            prior_pis[k] = 0.0
            continue

        Xk = X_source[idx_k, :]
        prior_mus[k, :] = Xk.mean(axis=0)
        prior_pis[k] = idx_k.size / n_samples

        if diag_only:
            var_k = Xk.var(axis=0, ddof=1)
            cov_k = np.diag(var_k + shrinkage)
        else:
            # Full empirical covariance with ridge
            cov_k = np.cov(Xk, rowvar=False)
            cov_k = cov_k + shrinkage * np.eye(d)

        prior_Sigmas[k, :, :] = cov_k

    # Normalize π to sum to 1 (should already be true, but just in case)
    if prior_pis.sum() > 0:
        prior_pis = prior_pis / prior_pis.sum()

    if verbose:
        print("[Priors] Done. Shapes:")
        print(f"  prior_mus:    {prior_mus.shape}")
        print(f"  prior_Sigmas: {prior_Sigmas.shape}")
        print(f"  prior_pis:    {prior_pis.shape}")

    return prior_mus, prior_Sigmas, prior_pis


# ----------------------------------------------------------------------
# 3. Construct structural A matrix from pathway / group annotations
# ----------------------------------------------------------------------

from typing import Union, List

def build_structural_A_from_pathways(
    feature_names: np.ndarray,
    pathway_info: Union[pd.DataFrame, Dict[str, List[str]]],
    feature_col: str = "gene",
    pathway_col: str = "pathway",
    min_group_size: int = 2,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build a structural adjacency mask A (d x d) from pathway/group annotations.

    A[i, j] = 1 if:
      - i == j (diagonal), OR
      - feature i and feature j co-occur in at least one pathway/group

    This A can be passed to Praxis_BGM as `sparse_A` (global mask). Internally,
    Praxis_BGM will convert it to jnp.array and tile over K clusters.

    Parameters
    ----------
    feature_names : array-like of str, shape (d,)
        Features actually used in the model (e.g., feature_names_used returned
        by prepare_source_target_datasets). Order defines the axes of A.
    pathway_info : DataFrame or dict
        If DataFrame:
            Must contain columns [pathway_col, feature_col], e.g. "pathway", "gene".
            Each row is (pathway_name, gene_name).
        If dict:
            Keys: pathway names (str)
            Values: list of genes for that pathway.
    feature_col : str, default="gene"
        Column name in pathway_info DataFrame containing feature/gene IDs.
    pathway_col : str, default="pathway"
        Column name in pathway_info DataFrame containing pathway names.
    min_group_size : int, default=2
        Minimum number of model features within a pathway required to form
        edges. Groups with fewer than this many overlapping features are ignored.
    verbose : bool, default=True
        If True, print summary information.

    Returns
    -------
    A : ndarray of shape (d, d)
        Structural adjacency mask:
          - A is symmetric
          - A[i, i] = 1 for all i
          - A[i, j] = 1 if features i and j share a pathway (and group >= min_group_size)
    """
    feature_names = np.asarray(feature_names)
    d = feature_names.size

    # Map feature -> index for quick lookup
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}

    # Normalize pathway information to a dict[pathway] -> list of features
    if isinstance(pathway_info, pd.DataFrame):
        if pathway_col not in pathway_info.columns or feature_col not in pathway_info.columns:
            raise ValueError(
                f"DataFrame must contain columns '{pathway_col}' and '{feature_col}'."
            )
        pathway_dict: Dict[str, List[str]] = {}
        for pw, gene in pathway_info[[pathway_col, feature_col]].itertuples(index=False):
            pathway_dict.setdefault(pw, []).append(str(gene))
    elif isinstance(pathway_info, dict):
        # Assume correct format
        pathway_dict = {str(k): [str(g) for g in v] for k, v in pathway_info.items()}
    else:
        raise TypeError("pathway_info must be a pandas DataFrame or dict.")

    # Initialize A as identity (self-connections always allowed)
    A = np.eye(d, dtype=np.float32)

    n_groups_used = 0
    n_pairs_added = 0
    features_with_any_group = np.zeros(d, dtype=bool)

    for pw_name, genes in pathway_dict.items():
        # Keep only genes that are present in feature_names
        idx_list = [feat_to_idx[g] for g in genes if g in feat_to_idx]

        if len(idx_list) < min_group_size:
            # Not enough overlapping features to form structural edges
            continue

        idx_array = np.array(idx_list, dtype=int)
        n_groups_used += 1
        features_with_any_group[idx_array] = True

        # Make all pairwise connections within this group
        # (i.e., complete subgraph / block structure)
        for i in idx_array:
            for j in idx_array:
                if i == j:
                    continue
                if A[i, j] == 0.0:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
                    n_pairs_added += 1

    if verbose:
        n_covered = int(features_with_any_group.sum())
        print("[A-Matrix] Built structural A from pathway annotations.")
        print(f"  - d (features):          {d}")
        print(f"  - Groups used:           {n_groups_used}")
        print(f"  - Features in any group: {n_covered}")
        print(f"  - Off-diagonal edges:    {n_pairs_added}")

    return A
'''
# ----------------------------------------------------------------------
# (Optional) Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Small mock example to show usage; you can delete this block if you want.
    rng = np.random.default_rng(0)

    # Fake source and target with overlapping + extra features
    n_s, n_t, d = 100, 80, 50
    src_features = [f"g{i}" for i in range(d)]
    tgt_features = [f"g{i}" for i in range(10, d + 10)]  # shifted names

    X_source_raw = rng.normal(size=(n_s, d))
    X_target_raw = rng.normal(size=(n_t, d))

    source_df = pd.DataFrame(X_source_raw, columns=src_features)
    target_df = pd.DataFrame(X_target_raw, columns=tgt_features)

    # Align + QC + select top-20 most variable, standardize
    Xs, Xt, feats, meta = prepare_source_target_datasets(
        source_df,
        target_df,
        top_n=20,
        standardize=True,
        verbose=True,
    )

    # Fake cluster labels on source
    labels = rng.integers(low=0, high=3, size=n_s)

    prior_mus, prior_Sigmas, prior_pis = build_gaussian_priors_from_source(
        Xs, labels, diag_only=False, verbose=True
    )

    print("Example finished.")
'''