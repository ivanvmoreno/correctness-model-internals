import pandas as pd
import pytest
import torch as pt

from src.classifying.activations_handler import (
    ActivationsHandler,
    combine_activations_handlers,
)


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


def test_sample_equally_across_groups_interleave(activations_handler):
    sampled = activations_handler.sample_equally_across_groups(
        [True, False], random=False, interleave=True
    )

    # Check we get equal numbers from each group
    true_count = sum(sampled.labels)
    false_count = sum(~sampled.labels)
    assert true_count == false_count

    # Check total size matches smallest group size * 2
    assert len(sampled.labels) == 10  # 5 from each group

    # Check interleaving
    for i in range(0, len(sampled.labels), 2):
        assert sampled.labels.iloc[i]  # True
        assert not sampled.labels.iloc[i + 1]  # False


def test_sample_equally_across_groups_interleave_random(activations_handler):
    # Test that random interleaved samples are different
    sample1 = activations_handler.sample_equally_across_groups(
        [True, False], random=True, interleave=True
    )
    sample2 = activations_handler.sample_equally_across_groups(
        [True, False], random=True, interleave=True
    )

    # Check we still get equal numbers from each group
    true_count = sum(sample1.labels)
    false_count = sum(~sample1.labels)
    assert true_count == false_count

    # Check total size matches smallest group size * 2
    assert len(sample1.labels) == 10  # 5 from each group

    # Check samples are different
    assert not pt.equal(sample1.activations, sample2.activations)

    # Check interleaving is still maintained
    for i in range(0, len(sample1.labels), 2):
        assert sample1.labels.iloc[i]  # True
        assert not sample1.labels.iloc[i + 1]  # False


def test_sample(activations_handler):
    # Test with fraction = 0.5
    sampled = activations_handler.sample(frac=0.5, random=False)
    assert len(sampled.labels) == 10
    assert sampled.activations.shape[0] == 10

    # Test with random=True
    sample1 = activations_handler.sample(frac=0.5, random=True)
    sample2 = activations_handler.sample(frac=0.5, random=True)
    assert not pt.equal(sample1.activations, sample2.activations)

    # Test with fraction = 1.0
    sampled = activations_handler.sample(frac=1.0, random=False)
    assert len(sampled.labels) == 20
    assert sampled.activations.shape[0] == 20
    assert pt.equal(sampled.activations, activations_handler.activations)


def test_add(activations_handler):
    # Split the handler into two parts
    part1, part2 = list(
        activations_handler.split_dataset([0.5, 0.5], random=False)
    )

    # Test adding them back together
    combined = part1 + part2
    pd.testing.assert_series_equal(combined.labels, activations_handler.labels)
    pt.testing.assert_close(
        combined.activations, activations_handler.activations
    )

    # Check labels match
    assert combined.labels.equals(activations_handler.labels)

    # Test adding empty handler raises error
    empty_handler = ActivationsHandler(
        activations=pt.tensor([]), labels=pd.Series([]), _allow_empty=True
    )
    with pytest.raises(ValueError):
        activations_handler + empty_handler


def test_init_edge_cases():
    # Test misaligned activations and labels
    with pytest.raises(ValueError):
        ActivationsHandler(
            activations=pt.tensor([[1.0, 2.0]]), labels=[True, False]
        )

    # Test empty activations and labels without _allow_empty
    with pytest.raises(ValueError):
        ActivationsHandler(activations=pt.tensor([]), labels=[])

    # Test empty activations and labels with _allow_empty
    handler = ActivationsHandler(
        activations=pt.tensor([]), labels=[], _allow_empty=True
    )
    assert len(handler.labels) == 0
    assert handler.activations.shape[0] == 0


def test_combine_activations_handlers(activations_handler):
    # Split into three parts
    parts = list(
        activations_handler.split_dataset([0.4, 0.3, 0.3], random=False)
    )

    # Test combining with equal_counts=False
    combined = combine_activations_handlers(parts)
    assert len(combined.labels) == len(activations_handler.labels)
    assert combined.activations.shape == activations_handler.activations.shape
    pd.testing.assert_series_equal(combined.labels, activations_handler.labels)
    pt.testing.assert_close(
        combined.activations, activations_handler.activations
    )

    # Test combining with equal_counts=True
    combined_equal = combine_activations_handlers(parts, equal_counts=True)
    min_count = min(len(part.labels) for part in parts)
    assert len(combined_equal.labels) == min_count * len(parts)
    # Verify each part has min_count samples
    for i in range(len(parts)):
        start_idx = i * min_count
        end_idx = (i + 1) * min_count
        pd.testing.assert_series_equal(
            combined_equal.labels[start_idx:end_idx].reset_index(drop=True),
            parts[i].labels[:min_count],
        )
        pt.testing.assert_close(
            combined_equal.activations[start_idx:end_idx],
            parts[i].activations[:min_count],
        )

    # Test combining single handler
    single_combined = combine_activations_handlers([parts[0]])
    assert pt.equal(single_combined.activations, parts[0].activations)
    assert single_combined.labels.equals(parts[0].labels)
