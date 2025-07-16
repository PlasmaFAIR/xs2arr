import numpy as np
from scipy.integrate import simpson


def compute_rate_trapezoid(
    energies: np.ndarray, cross_section: np.ndarray, eedf: np.ndarray
) -> float:
    return np.trapezoid(np.sqrt(energies) * cross_section * eedf, x=energies)


def compute_rate_simpson(
    energies: np.ndarray, cross_section: np.ndarray, eedf: np.ndarray
) -> float:
    return simpson(np.sqrt(energies) * cross_section * eedf, x=energies)
