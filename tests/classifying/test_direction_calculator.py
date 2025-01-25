import pytest
import torch as pt

from src.classifying.direction_calculator import DirectionCalculator


@pytest.fixture
def simple_activations():
    """Create simple activations for testing with imbalanced groups."""
    # 5 samples in 'from' group
    activations_from = pt.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [2.0, 3.0],
            [1.5, 2.5],
            [2.5, 3.5],
        ]
    )
    # 2 samples in 'to' group
    activations_to = pt.tensor(
        [
            [5.0, 6.0],
            [6.0, 7.0],
        ]
    )
    return activations_from, activations_to


def test_initialization(simple_activations):
    """Test that the DirectionCalculator initializes correctly."""
    activations_from, activations_to = simple_activations
    calculator = DirectionCalculator(
        activations_from, activations_to, balance=True
    )

    assert pt.allclose(calculator.centroid_from, pt.tensor([2.0, 3.0]))
    assert pt.allclose(calculator.centroid_to, pt.tensor([5.5, 6.5]))
    assert pt.allclose(calculator.max_activations_from, pt.tensor([3.0, 4.0]))
    assert pt.allclose(calculator.min_activations_from, pt.tensor([1.0, 2.0]))
    assert pt.allclose(calculator.max_activations_to, pt.tensor([6.0, 7.0]))
    assert pt.allclose(calculator.min_activations_to, pt.tensor([5.0, 6.0]))


def test_mean_activations_balanced(simple_activations):
    """Test mean_activations property with balanced=True."""
    activations_from, activations_to = simple_activations
    calculator = DirectionCalculator(
        activations_from, activations_to, balance=True
    )

    expected_mean = 0.5 * (pt.tensor([2.0, 3.0]) + pt.tensor([5.5, 6.5]))
    assert pt.allclose(calculator.mean_activations, expected_mean)


def test_mean_activations_unbalanced(simple_activations):
    """Test mean_activations property with balanced=False."""
    activations_from, activations_to = simple_activations
    calculator = DirectionCalculator(
        activations_from, activations_to, balance=False
    )

    # With 5 samples in 'from' and 2 in 'to', mean will be biased towards 'from'
    all_activations = pt.cat([activations_from, activations_to])
    expected_mean = all_activations.mean(dim=0)  # (5 * from + 2 * to) / 7
    assert pt.allclose(calculator.mean_activations, expected_mean)


def test_classifying_direction_balanced(simple_activations):
    """Test classifying_direction property with balanced=True."""
    activations_from, activations_to = simple_activations
    calculator = DirectionCalculator(
        activations_from, activations_to, balance=True
    )

    # Direction should point from 'from' group to 'to' group
    # Should be half of the difference between centroids
    assert pt.allclose(
        calculator.classifying_direction, pt.tensor([1.75, 1.75])
    )

    # When balanced, direction magnitude should not be affected by group sizes
    distances_from = calculator.get_distance_along_classifying_direction(
        activations_from
    )
    distances_to = calculator.get_distance_along_classifying_direction(
        activations_to
    )
    assert abs(distances_from.mean()) - abs(distances_to.mean()) < 1e-5


def test_classifying_direction_unbalanced(simple_activations):
    """Test classifying_direction property with balanced=False."""
    activations_from, activations_to = simple_activations
    calculator = DirectionCalculator(
        activations_from, activations_to, balance=False
    )

    direction = calculator.classifying_direction

    # Direction should point from 'from' group to 'to' group
    assert direction[0] > 0 and direction[1] > 0

    # When unbalanced, the larger group ('from') should have less distance on average
    distances_from = calculator.get_distance_along_classifying_direction(
        activations_from
    )
    distances_to = calculator.get_distance_along_classifying_direction(
        activations_to
    )
    assert abs(distances_from.mean()) < abs(distances_to.mean())


def test_classifying_direction_balanced_vs_unbalanced_same_size_groups():
    """Test that balanced and unbalanced give same result with equal group sizes."""
    # Create balanced groups with 3 samples each
    activations_from = pt.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    activations_to = pt.tensor([[4.0, 5.0], [5.0, 6.0], [6.0, 7.0]])

    # Calculate direction with balanced=True
    calculator_balanced = DirectionCalculator(
        activations_from, activations_to, balance=True
    )
    direction_balanced = calculator_balanced.classifying_direction

    # Calculate direction with balanced=False
    calculator_unbalanced = DirectionCalculator(
        activations_from, activations_to, balance=False
    )
    direction_unbalanced = calculator_unbalanced.classifying_direction

    # Directions should be identical since groups are already balanced
    assert pt.allclose(direction_balanced, direction_unbalanced)

    # Both should point from 'from' group to 'to' group
    assert direction_balanced[0] > 0 and direction_balanced[1] > 0


def test_get_distance_along_classifying_direction(simple_activations):
    """Test distance calculation along the classifying direction."""
    activations_from, activations_to = simple_activations
    calculator = DirectionCalculator(
        activations_from, activations_to, balance=True
    )

    # Test points from the 'from' group should have negative distances
    distances_from = calculator.get_distance_along_classifying_direction(
        activations_from
    )
    assert (distances_from < 0).all()

    # Test points from the 'to' group should have positive distances
    distances_to = calculator.get_distance_along_classifying_direction(
        activations_to
    )
    assert (distances_to > 0).all()

    # Test that centroids fall on the correct positions
    centroid_from_distance = (
        calculator.get_distance_along_classifying_direction(
            calculator.centroid_from.unsqueeze(0)
        )
    )
    centroid_to_distance = calculator.get_distance_along_classifying_direction(
        calculator.centroid_to.unsqueeze(0)
    )

    # In the balanced case, centroids should be equidistant from mean
    assert pt.allclose(
        centroid_from_distance, pt.tensor([-((2 * 1.75**2) ** 0.5)])
    )
    assert pt.allclose(centroid_to_distance, pt.tensor([(2 * 1.75**2) ** 0.5]))

    # Test that mean_activations is at zero
    mean_distance = calculator.get_distance_along_classifying_direction(
        calculator.mean_activations.unsqueeze(0)
    )
    assert pt.allclose(mean_distance, pt.tensor([0.0]))


def test_single_sample_groups():
    """Test behavior when groups have single samples."""
    activations_from = pt.tensor([[1.0, 2.0]])
    activations_to = pt.tensor([[5.0, 6.0]])
    calculator = DirectionCalculator(
        activations_from, activations_to, balance=True
    )

    # Direction should still point correctly
    direction = calculator.classifying_direction
    assert direction[0] > 0 and direction[1] > 0

    # Distances should still be correctly signed
    distances_from = calculator.get_distance_along_classifying_direction(
        activations_from
    )
    distances_to = calculator.get_distance_along_classifying_direction(
        activations_to
    )
    assert (distances_from < 0).all()
    assert (distances_to > 0).all()


def test_edge_cases():
    """Test edge cases with different dimensionality."""
    # Test with single-dimensional data
    activations_from = pt.tensor([[1.0], [2.0]])
    activations_to = pt.tensor([[3.0], [4.0]])
    calculator = DirectionCalculator(
        activations_from, activations_to, balance=True
    )
    assert calculator.classifying_direction.shape == (1,)

    # Test with higher-dimensional data
    activations_from = pt.randn(10, 5)
    activations_to = pt.randn(15, 5)
    calculator = DirectionCalculator(
        activations_from, activations_to, balance=True
    )
    assert calculator.classifying_direction.shape == (5,)
