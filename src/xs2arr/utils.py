import numpy as np

# CODATA 2022 values from Rev. Mod. Phys. 97, 025002.
q = 1.602176634e-19  # Elementary charge
m_e = 9.1093837139e-31  # Electron mass

_log10_e = np.log10(np.exp(1.0))


def arrhenius(T, a, b, c):
    return a * T**b * np.exp(c / T)


def arrhenius_log(T, log10_a, b, c):
    return log10_a + b * np.log10(T) + _log10_e * c / T
