from numpy import exp

# CODATA 2022 values from Rev. Mod. Phys. 97, 025002.
q = 1.602176634e-19  # Elementary charge
m_e = 9.1093837139e-31  # Electron mass


def arrhenius(T, a, b, c):
    return a * T**b * exp(c / T)
