import pandas as pd
import pytest
import torch as pt

from src.classifying.activations_handler import ActivationsHandler


@pytest.fixture
def sample_activations():
    # Create 20 samples with 3 features each
    return pt.tensor(
        [[float(i), float(i + 1), float(i + 2)] for i in range(0, 60, 3)]
    )


@pytest.fixture
def sample_labels():
    # Create imbalanced dataset: 15 True, 5 False
    return [True] * 15 + [False] * 5


@pytest.fixture
def activations_handler(sample_activations, sample_labels):
    return ActivationsHandler(
        activations=sample_activations, labels=sample_labels
    )


def test_init(sample_activations, sample_labels):
    handler = ActivationsHandler(
        activations=sample_activations, labels=sample_labels
    )
    assert pt.equal(handler.activations, sample_activations)
    assert handler.labels.equals(pd.Series(sample_labels))
    assert len(handler.labels) == 20
    assert sum(handler.labels) == 15  # Check imbalance


def test_get_groups(activations_handler):
    # Test single label
    true_group = activations_handler.get_groups(True)
    assert true_group.activations.shape[0] == 15
    assert len(true_group.labels) == 15
    assert true_group.labels.all()

    false_group = activations_handler.get_groups(False)
    assert false_group.activations.shape[0] == 5
    assert len(false_group.labels) == 5
    assert not false_group.labels.any()

    # Test multiple labels
    both_groups = activations_handler.get_groups([True, False])
    assert both_groups.activations.shape[0] == 20
    assert len(both_groups.labels) == 20

    # Test invalid label
    with pytest.raises(ValueError):
        activations_handler.get_groups("invalid")

    with pytest.raises(ValueError):
        activations_handler.get_groups([True, "invalid"])


def test_shuffle(activations_handler):
    shuffled = activations_handler.shuffle()

    # Check shapes remain same
    assert shuffled.activations.shape == activations_handler.activations.shape
    assert len(shuffled.labels) == len(activations_handler.labels)

    # Check data is actually shuffled but contains same elements
    assert not pt.equal(shuffled.activations, activations_handler.activations)
    assert set(shuffled.labels) == set(activations_handler.labels)

    # Check class distribution remains the same
    assert sum(shuffled.labels) == 15
    assert sum(~shuffled.labels) == 5


@pytest.fixture(params=[True, False])
def shuffle(request):
    return request.param


def test_split_dataset(activations_handler, shuffle):
    # Test 50-50 split
    splits = list(activations_handler.split_dataset([1, 1], random=shuffle))
    assert len(splits) == 2
    assert splits[0].activations.shape[0] == 10
    assert splits[1].activations.shape[0] == 10

    # Verify class distribution in splits
    assert (
        sum(splits[0].labels) + sum(splits[1].labels) == 15
    )  # Total True labels

    # Test uneven split (60-40)
    splits = list(activations_handler.split_dataset([0.6, 0.4], random=shuffle))
    assert len(splits) == 2
    assert splits[0].activations.shape[0] == 12  # 60% of 20
    assert splits[1].activations.shape[0] == 8  # 40% of 20

    # Test three-way split
    splits = list(
        activations_handler.split_dataset([0.5, 0.3, 0.2], random=shuffle)
    )
    assert len(splits) == 3
    assert splits[0].activations.shape[0] == 10  # 50% of 20
    assert splits[1].activations.shape[0] == 6  # 30% of 20
    assert splits[2].activations.shape[0] == 4  # 20% of 20

    # Test invalid splits
    with pytest.raises(ValueError):
        list(activations_handler.split_dataset([], random=shuffle))

    with pytest.raises(ValueError):
        list(activations_handler.split_dataset([-1, 1], random=shuffle))

    if shuffle:
        # Only test randomness when random=True
        splits1 = list(activations_handler.split_dataset([1, 1], random=True))
        splits2 = list(activations_handler.split_dataset([1, 1], random=True))

        assert not pt.equal(splits1[0].activations, splits2[0].activations)
        assert not pt.equal(splits1[1].activations, splits2[1].activations)


def test_sample_equally_across_groups(activations_handler, shuffle):
    sampled = activations_handler.sample_equally_across_groups(
        [True, False], random=shuffle
    )

    # Check we get equal numbers from each group
    true_count = sum(sampled.labels)
    false_count = sum(~sampled.labels)
    assert true_count == false_count

    # Check total size matches smallest group size * 2
    assert (
        len(sampled.labels) == 10
    )  # 5 from each group (limited by False group size)

    if shuffle:
        # Only test randomness when random=True
        sample1 = activations_handler.sample_equally_across_groups(
            [True, False], random=True
        )
        sample2 = activations_handler.sample_equally_across_groups(
            [True, False], random=True
        )
        assert not pt.equal(sample1.activations, sample2.activations)
