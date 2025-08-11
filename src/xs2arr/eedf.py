from math import gamma

import numpy as np
from numpy.typing import ArrayLike

from xs2arr.utils import validate_real


# EEDF constants - Plasma Sources Sci. Technol. 27 095008 2018 equations 4 and 5 (except mean energy terms).
def _beta1(x: float):
    return x * ((gamma(2.5 / x)) ** 1.5) / ((gamma(1.5 / x)) ** 2.5)


def _beta2(x: float):
    return (gamma(2.5 / x) / gamma(1.5 / x)) ** x


class EEDF:
    def __init__(
        self,
        g: float | int,
        Te: float | int | None = None,
        mean_E: float | int | None = None,
    ) -> None:
        g = validate_real(g, "g", lambda x: 1.0 <= x <= 2.0)

        if Te is None and mean_E is None:
            raise ValueError("Either Te or mean_E must be provided")
        if Te is not None and mean_E is not None:
            raise ValueError("Only one of Te or mean_E must be provided")
        if mean_E is not None:
            mean_E = validate_real(mean_E, "mean_E", lambda x: x > 0.0)
            Te = 2.0 * mean_E / 3.0
        Te = validate_real(Te, "Te", lambda x: x > 0.0)

        self._g = g
        self._Te = Te
        self._mean_E = 3.0 * Te / 2.0

        self._beta1 = _beta1(g)
        self._beta2 = _beta2(g)

    def pdf(self, energies: ArrayLike) -> np.ndarray:
        """
        Calculate the electron energy distribution function (EEDF) probability density.

        Parameters
        ----------
        energies
            Energy values at which to compute the EEDF probability density.
        """
        energies = np.asarray(energies, dtype=float)

        return (
            self._beta1
            * np.sqrt(energies)
            * np.exp(-(self._beta2 * energies**self.g / self.mean_E))
            / self.mean_E**1.5
        )

    @property
    def g(self) -> float:
        return self._g

    @property
    def Te(self) -> float:
        return self._Te

    @property
    def mean_E(self) -> float:
        return self._mean_E

    @property
    def beta1(self) -> float:
        return self._beta1

    @property
    def beta2(self) -> float:
        return self._beta2


class Maxwellian(EEDF):
    def __init__(
        self, Te: float | int | None = None, mean_E: float | int | None = None
    ):
        super().__init__(g=1.0, Te=Te, mean_E=mean_E)

    def pdf(self, energies: ArrayLike) -> np.ndarray:
        energies = np.asarray(energies, dtype=float)

        return (
            self._beta1
            * np.sqrt(energies)
            * np.exp(-self._beta2 * energies / self.mean_E)
            / self.mean_E**1.5
        )


class Druyvesteyn(EEDF):
    def __init__(
        self, Te: float | int | None = None, mean_E: float | int | None = None
    ):
        super().__init__(g=2.0, Te=Te, mean_E=mean_E)
