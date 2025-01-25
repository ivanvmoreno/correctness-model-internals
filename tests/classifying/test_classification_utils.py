import numpy as np
import pandas as pd
import pytest
import torch as pt
from sklearn.metrics import accuracy_score, f1_score

from classifying.classification_utils import BinaryClassifier


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

    # Without explicit classification_cut, optimal_cut should equal optimal_train_set_cut
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
