import numpy as np
import pandas as pd
import pytest

import praxis_bgm.prior_utils as prior_utils


def test_prepare_source_target_datasets_aligns_standardizes_and_selects_top_features():
    source = pd.DataFrame(
        {
            "g1": [1.0, 2.0, 3.0, np.nan],
            "g2": [5.0, 5.0, 5.0, 5.0],
            "g3": [0.0, 1.0, 2.0, 3.0],
            "g4": [10.0, 11.0, 9.0, 12.0],
        }
    )
    target = pd.DataFrame(
        {
            "g0": [0.0, 1.0, 2.0, 3.0],
            "g1": [2.0, 3.0, 4.0, 5.0],
            "g3": [1.0, 2.0, 3.0, 4.0],
            "g4": [11.0, 10.0, 12.0, 13.0],
        }
    )

    Xs, Xt, features, metadata = prior_utils.prepare_source_target_datasets(
        source,
        target,
        top_n=2,
        standardize=True,
        impute_strategy="mean",
        min_non_null_fraction=0.5,
        verbose=False,
    )

    assert Xs.shape == (4, 2)
    assert Xt.shape == (4, 2)
    assert features.shape == (2,)
    assert set(features).issubset({"g1", "g3", "g4"})
    assert "g2" not in features
    np.testing.assert_allclose(Xs.mean(axis=0), np.zeros(2), atol=1e-6)
    assert metadata["scaler"] is not None
    assert metadata["imputer"] is not None


def test_prepare_source_target_datasets_supports_array_inputs_with_sample_axis_one():
    source = np.array([[1.0, 2.0, 3.0], [10.0, 12.0, 14.0]], dtype=np.float32)
    target = np.array([[4.0, 5.0, 6.0], [11.0, 13.0, 15.0]], dtype=np.float32)

    Xs, Xt, features, metadata = prior_utils.prepare_source_target_datasets(
        source,
        target,
        source_feature_names=np.array(["a", "b"]),
        target_feature_names=np.array(["a", "b"]),
        sample_axis=1,
        top_n=None,
        standardize=False,
        verbose=False,
    )

    assert Xs.shape == (3, 2)
    assert Xt.shape == (3, 2)
    np.testing.assert_array_equal(features, np.array(["a", "b"]))
    assert metadata["scaler"] is None


def test_prepare_source_target_datasets_rejects_nonoverlapping_features():
    source = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    target = pd.DataFrame({"x": [5.0, 6.0], "y": [7.0, 8.0]})

    with pytest.raises(ValueError, match="No overlapping features"):
        prior_utils.prepare_source_target_datasets(source, target, verbose=False)


def test_build_gaussian_priors_from_source_returns_valid_shapes_and_weights():
    X = np.array(
        [
            [0.0, 1.0],
            [0.2, 1.1],
            [3.0, 4.0],
            [3.2, 4.1],
            [6.0, 7.0],
            [6.1, 7.2],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1, 2, 2])

    mus, Sigmas, pis = prior_utils.build_gaussian_priors_from_source(
        X,
        labels,
        diag_only=False,
        shrinkage=1e-5,
        verbose=False,
    )

    assert mus.shape == (3, 2)
    assert Sigmas.shape == (3, 2, 2)
    assert pis.shape == (3,)
    assert pis.sum() == pytest.approx(1.0)
    assert np.all(np.linalg.eigvalsh(Sigmas) > 0)


def test_build_gaussian_priors_from_source_diag_only_returns_diagonal_covariances():
    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [0.1, 1.1, 2.2],
            [3.0, 4.0, 5.0],
            [2.9, 4.2, 4.8],
        ],
        dtype=np.float32,
    )
    labels = np.array(["a", "a", "b", "b"])

    _, Sigmas, pis = prior_utils.build_gaussian_priors_from_source(
        X,
        labels,
        diag_only=True,
        verbose=False,
    )

    assert pis.sum() == pytest.approx(1.0)
    off_diag = Sigmas - np.array([np.diag(np.diag(s)) for s in Sigmas])
    np.testing.assert_allclose(off_diag, np.zeros_like(off_diag), atol=1e-8)


def test_build_structural_A_from_pathways_connects_features_with_shared_groups():
    feature_names = np.array(["g1", "g2", "g3", "g4"])
    pathway_info = {
        "p1": ["g1", "g2"],
        "p2": ["g2", "g3"],
        "p3": ["outside_only"],
    }

    A = prior_utils.build_structural_A_from_pathways(
        feature_names,
        pathway_info,
        min_group_size=2,
        verbose=False,
    )

    assert A.shape == (4, 4)
    np.testing.assert_allclose(A, A.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(A), np.ones(4), atol=1e-6)
    assert A[0, 1] == pytest.approx(1.0)
    assert A[1, 2] == pytest.approx(1.0)
    assert A[0, 3] == pytest.approx(0.0)
