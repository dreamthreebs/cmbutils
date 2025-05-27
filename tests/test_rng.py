import numpy as np
import secrets

from cmbutils.rng import generate_seeds


def check_unique_seeds(seeds: np.ndarray) -> bool:
    """Check if all elements in the array are unique."""
    seen = set()
    for seed in seeds:
        if seed in seen:
            return False
        seen.add(seed)
    return True


def test_generated_seeds():
    num_seeds = 2000

    seeds = generate_seeds(num_seeds)
    assert check_unique_seeds(seeds), "Duplicate seeds found!"
    print("All seeds are unique.")
