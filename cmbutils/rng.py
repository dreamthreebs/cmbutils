import numpy as np
import secrets


def generate_seeds(num_seeds: int) -> np.ndarray:
    """Generate unique seeds, save to file, and return them as np.ndarray."""
    seeds = [secrets.randbits(32) for _ in range(num_seeds)]
    return np.array(seeds)
