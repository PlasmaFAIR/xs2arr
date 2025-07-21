import numpy as np
from numpy.typing import ArrayLike

# CODATA 2022 values from Rev. Mod. Phys. 97, 025002.
q = 1.602176634e-19  # Elementary charge
m_e = 9.1093837139e-31  # Electron mass

_log10_e = np.log10(np.exp(1.0))


def arrhenius(T: ArrayLike, a: float, b: float, c: float) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    return a * T**b * np.exp(c / T)


def arrhenius_log(T: ArrayLike, log10_a: float, b: float, c: float) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    return log10_a + b * np.log10(T) + _log10_e * c / T
