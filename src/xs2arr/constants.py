from math import gamma

# CODATA 2022 values from Rev. Mod. Phys. 97, 025002.
q = 1.602176634e-19  # Elementary charge
m_e = 9.1093837139e-31  # Electron mass


# EEDF constants.
def _beta1(x: float):
    return gamma(2.5 / x) ** 1.5 * gamma(1.5 / x) ** -2.5


def _beta2(x: float):
    return gamma(2.5 / x) / gamma(1.5 / x)


maxwellian_beta1 = _beta1(x=1.0)
maxwellian_beta2 = _beta2(x=1.0)
druyvesteyn_beta1 = _beta1(x=2.0)
druyvesteyn_beta2 = _beta2(x=2.0)
