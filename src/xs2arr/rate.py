from numpy import ndarray, sqrt, trapezoid

from xs2arr.utils import m_e, q


def compute_rate(energies: ndarray, cross_section: ndarray, eedf: ndarray) -> float:
    return sqrt(2.0 * q / m_e) * trapezoid(
        sqrt(energies) * cross_section * eedf, x=energies
    )
