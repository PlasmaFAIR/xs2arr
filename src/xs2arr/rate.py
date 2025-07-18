import numpy as np
from scipy.integrate import simpson


def compute_rate_trapezoid(
    energies: np.ndarray, cross_section: np.ndarray, eedf: np.ndarray
) -> float:
    """
    Compute reaction rate using trapezoidal integration.

    Parameters
    ----------
    energies : np.ndarray
        Array of energy values
    cross_section : np.ndarray
        Array of cross section values corresponding to energies
    eedf : np.ndarray
        Array of electron energy distribution function values

    Returns
    -------
    float
        Reaction rate computed using trapezoidal integration
    """

    return np.trapezoid(np.sqrt(energies) * cross_section * eedf, x=energies)


def compute_rate_simpson(
    energies: np.ndarray, cross_section: np.ndarray, eedf: np.ndarray
) -> float:
    """
    Compute reaction rate using Simpson's integration.

    Parameters
    ----------
    energies : np.ndarray
        Array of energy values
    cross_section : np.ndarray
        Array of cross section values corresponding to energies
    eedf : np.ndarray
        Array of electron energy distribution function values

    Returns
    -------
    float
        Reaction rate computed using Simpson's integration
    """

    return simpson(np.sqrt(energies) * cross_section * eedf, x=energies)
