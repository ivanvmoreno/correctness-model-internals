import numpy as np
import pandas as pd
import pytest
import torch as pt
from sklearn.metrics import accuracy_score, f1_score

from classifying.activations_handler import ActivationsHandler
from classifying.classification_utils import (
    BinaryClassifier,
    get_between_class_variance_and_within_class_variance,
)


@pytest.fixture
def sample_classification_data():
    """Create sample data for testing BinaryClassifier."""
    # Create training data
    train_labels = pd.Series([False, False, True, True, True], dtype=bool)
    train_scores = pt.tensor([-2.0, -1.0, 0.7, 1.0, 2.0])

    # Create test data
    test_labels = pd.Series([False, False, True, True], dtype=bool)
    test_scores = pt.tensor([-1.5, -0.5, 1.5, 2.5])

    return train_labels, train_scores, test_labels, test_scores


def test_binary_classifier_metrics(sample_classification_data):
    """Test that classification metrics are calculated correctly."""
    train_labels, train_scores, test_labels, test_scores = (
        sample_classification_data
    )

    classifier = BinaryClassifier(
        train_labels, train_scores, test_labels, test_scores
    )

    # ROC AUC should be between 0 and 1
    assert 0 <= classifier.test_roc_auc <= 1

    # Test predictions with custom classification cut
    classifier_with_cut = BinaryClassifier(
        train_labels,
        train_scores,
        test_labels,
        test_scores,
        classification_cut=0.0,
    )

    assert pt.allclose(
        classifier_with_cut.test_pred_class,
        pt.tensor([False, False, True, True]),
    )

    # Test with explicit classification cut
    classifier_with_cut = BinaryClassifier(
        train_labels,
        train_scores,
        test_labels,
        test_scores,
        classification_cut=-1.0,
    )

    assert pt.allclose(
        classifier_with_cut.test_pred_class,
        pt.tensor([False, True, True, True]),
    )


def test_binary_classifier_invalid_inputs():
    """Test that BinaryClassifier raises appropriate errors for invalid inputs."""
    # Test non-boolean labels
    with pytest.raises(TypeError, match="Labels must be boolean"):
        BinaryClassifier(
            pd.Series([0, 1, 1]),  # Integer labels instead of boolean
            pt.tensor([-1.0, 0.0, 1.0]),
            pd.Series([0, 1]),
            pt.tensor([-0.5, 0.5]),
        )


def test_binary_classifier_custom_metrics(sample_classification_data):
    """Test BinaryClassifier with custom metric functions."""
    train_labels, train_scores, test_labels, test_scores = (
        sample_classification_data
    )

    classifier = BinaryClassifier(
        train_labels,
        train_scores,
        test_labels,
        test_scores,
        classification_metric_funcs=(accuracy_score, f1_score),
    )

    # Verify that custom metrics are stored
    assert len(classifier.classification_metric_funcs) == 2
    assert classifier.classification_metric_funcs[0] == accuracy_score
    assert classifier.classification_metric_funcs[1] == f1_score


def test_binary_classifier_roc_curve(sample_classification_data):
    """Test that ROC curve values are properly calculated."""
    train_labels, train_scores, test_labels, test_scores = (
        sample_classification_data
    )

    classifier = BinaryClassifier(
        train_labels, train_scores, test_labels, test_scores
    )

    # Check that FPR and TPR have same length and are monotonic
    assert len(classifier.train_roc_fprs) == len(classifier.train_roc_tprs)
    assert np.all(np.diff(classifier.train_roc_fprs) >= 0)
    assert np.all(np.diff(classifier.train_roc_tprs) >= 0)

    # Check that FPR and TPR are between 0 and 1
    assert np.all(classifier.train_roc_fprs >= 0)
    assert np.all(classifier.train_roc_fprs <= 1)
    assert np.all(classifier.train_roc_tprs >= 0)
    assert np.all(classifier.train_roc_tprs <= 1)


def test_binary_classifier_optimal_cut(sample_classification_data):
    """Test that optimal cut calculation works correctly."""
    train_labels, train_scores, test_labels, test_scores = (
        sample_classification_data
    )

    classifier = BinaryClassifier(
        train_labels, train_scores, test_labels, test_scores
    )

    # Check that optimal cut is a float
    assert isinstance(classifier.optimal_cut, float)
    assert isinstance(classifier.optimal_train_set_cut, float)

    assert np.isclose(classifier.optimal_train_set_cut, 0.7)

    # Without explicit classification_cut, optimal_cut should equal
    # optimal_train_set_cut
    assert np.isclose(classifier.optimal_cut, classifier.optimal_train_set_cut)

    # Test with explicit classification cut
    explicit_cut = 0.0
    classifier_with_cut = BinaryClassifier(
        train_labels,
        train_scores,
        test_labels,
        test_scores,
        classification_cut=explicit_cut,
    )
    assert np.isclose(classifier_with_cut.optimal_cut, explicit_cut)


def test_between_and_within_class_variance():
    """Test calculation of between and within class variance."""
    # Test case where all activations are identical within each class
    identical_activations = pt.tensor(
        [
            [1.0, 1.0],  # Class True
            [1.0, 1.0],  # Class True
            [-1.0, -1.0],  # Class False
            [-1.0, -1.0],  # Class False
        ]
    )
    identical_labels = pd.Series([True, True, False, False])
    ah_identical = ActivationsHandler(identical_activations, identical_labels)

    between_var, within_var = (
        get_between_class_variance_and_within_class_variance(ah_identical)
    )

    # Within class variance should be 0 since activations are identical within classes
    assert within_var < 1e-8
    # Between class variance should be 1.0 since classes are different by 2 units
    # Global mean = (1 + 1 + (-1) + (-1))/4 = 0
    # Group means: True = [1,1], False = [-1,-1]
    # Average distance from each group mean to global mean = ((1-0)^2 + (1-0)^2) / 2 = 1
    assert np.isclose(between_var, 1.0)  # Average distance = 1

    # Test case where all activations are identical across classes
    same_activations = pt.tensor(
        [
            [1.0, 1.0],  # Class True
            [1.0, 1.0],  # Class True
            [1.0, 1.0],  # Class False
            [1.0, 1.0],  # Class False
        ]
    )
    same_labels = pd.Series([True, True, False, False])
    ah_same = ActivationsHandler(same_activations, same_labels)

    between_var, within_var = (
        get_between_class_variance_and_within_class_variance(ah_same)
    )

    # Between class variance should be 0 since classes have same mean
    assert between_var < 1e-8
    # Within class variance should be 0 since all activations are identical
    assert within_var < 1e-8

    # Test case where all activations are different but class means are identical
    diff_activations = pt.tensor(
        [
            [2.0, 2.0],  # Class True
            [0.0, 0.0],  # Class True
            [1.5, 1.5],  # Class False
            [0.5, 0.5],  # Class False
        ]
    )
    diff_labels = pd.Series([True, True, False, False])
    ah_diff = ActivationsHandler(diff_activations, diff_labels)

    between_var, within_var = (
        get_between_class_variance_and_within_class_variance(ah_diff)
    )

    # Between class variance should be 0 since class means are both [1,1]
    assert between_var < 1e-8
    # Within class variance should be 0.625
    # True class: points at [2,2] and [0,0], mean at [1,1], contributes 1.0
    # False class: points at [1.5,1.5] and [0.5,0.5], mean at [1,1], contributes 0.25
    # Average within class variance = (1.0 + 0.25) / 2 = 0.625
    assert np.isclose(within_var, 0.625)

    # Test case where all activations are different and class means are different
    diff_means_activations = pt.tensor(
        [
            [3.0, 3.0],  # Class True
            [1.0, 1.0],  # Class True
            [-1.0, -1.0],  # Class False
            [-3.0, -3.0],  # Class False
        ]
    )
    diff_means_labels = pd.Series([True, True, False, False])
    ah_diff_means = ActivationsHandler(
        diff_means_activations, diff_means_labels
    )

    between_var, within_var = (
        get_between_class_variance_and_within_class_variance(ah_diff_means)
    )

    # Between class variance should be 4.0
    # Global mean = [0,0]
    # True class mean = [2,2], False class mean = [-2,-2]
    # Average distance from each group mean to global mean = ((2-0)^2 + (2-0)^2) / 2 = 4.0
    assert np.isclose(between_var, 4.0)
    # Within class variance should be 1.0
    # Each class has points ±2 from its mean
    # Each class contributes: 1.0 per dimension to within class variance
    # True class: points at 3,1 with mean 2 in each dim -> variance 1.0 per dim
    # False class: points at -1,-3 with mean -2 in each dim -> variance 1.0 per dim
    # Average within class variance = (1.0 + 1.0) / 2 = 1.0 per dim
    # Total variance averaged across 2 dims = 1.0
    assert np.isclose(within_var, 1.0)

    # Test case with different x,y values but similar structure
    diff_xy_activations = pt.tensor(
        [
            [2.5, 3.5],  # Class True
            [1.5, 0.5],  # Class True
            [-0.5, -1.5],  # Class False
            [-3.5, -2.5],  # Class False
        ]
    )
    diff_xy_labels = pd.Series([True, True, False, False])
    ah_diff_xy = ActivationsHandler(diff_xy_activations, diff_xy_labels)

    between_var, within_var = (
        get_between_class_variance_and_within_class_variance(ah_diff_xy)
    )

    # Between class variance should be 4.0
    # Global mean = [0,0]
    # True class mean = [2,2], False class mean = [-2,-2]
    # Average distance from each group mean to global mean = ((2-0)^2 + (2-0)^2) / 2 = 4.0
    assert np.isclose(between_var, 4.0)
    # Within class variance should be 1.25
    # Each class has points that average to ±2 from global mean
    # True class: points at [2.5,3.5] and [1.5,0.5] with mean [2,2] -> variance 0.25 per dim
    # False class: points at [-0.5,-1.5] and [-3.5,-2.5] with mean [-2,-2] -> variance 2.25 and 0.25 per dim
    # Average within class variance = (2.25 + 0.25) / 2 = 1.25 per dim
    # Total variance averaged across 2 dims = 1.25
    assert np.isclose(within_var, 1.25)
