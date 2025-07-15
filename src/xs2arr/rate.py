import numpy as np


def compute_rate(
    energies: np.ndarray, cross_section: np.ndarray, eedf: np.ndarray
) -> float:
    return np.trapezoid(np.sqrt(energies) * cross_section * eedf, x=energies)
