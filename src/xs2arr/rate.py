from numpy import ndarray, sqrt, trapezoid


def compute_rate(energies: ndarray, cross_section: ndarray, eedf: ndarray) -> float:
    return trapezoid(sqrt(energies) * cross_section * eedf, x=energies)
