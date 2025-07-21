from abc import ABC, abstractmethod
from math import gamma

import numpy as np
from numpy.typing import ArrayLike


# EEDF constants - Plasma Sources Sci. Technol. 27 095008 2018 equations 4 and 5 (except mean energy terms).
def _beta1(x: float):
    return x * ((gamma(2.5 / x)) ** 1.5) / ((gamma(1.5 / x)) ** 2.5)


def _beta2(x: float):
    return (gamma(2.5 / x) / gamma(1.5 / x)) ** x


_maxwellian_beta1 = _beta1(x=1.0)
_maxwellian_beta2 = _beta2(x=1.0)
_druyvesteyn_beta1 = _beta1(x=2.0)
_druyvesteyn_beta2 = _beta2(x=2.0)


class EEDF(ABC):
    def __init__(
        self, g: float, Te: float | None = None, mean_E: float | None = None
    ) -> None:
        if not isinstance(g, float):
            raise TypeError("g must be of type float")
        if not (1.0 <= g <= 2.0):
            raise ValueError("g must be in the range 1.0 <= g <= 2.0")
        if Te is None and mean_E is None:
            raise ValueError("Either Te or mean_E must be provided")
        if Te is not None and mean_E is not None:
            raise ValueError("Only one of Te or mean_E must be provided")
        if mean_E is not None:
            if not isinstance(mean_E, float):
                raise TypeError("mean_E must be of type float")
            Te = 2.0 * mean_E / 3.0
        if not isinstance(Te, float):
            raise TypeError("Te must be of type float")

        self._g = g
        self._Te = Te
        self._mean_E = 3.0 * Te / 2.0

        self._beta1 = _beta1(g)
        self._beta2 = _beta2(g)

    @abstractmethod
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
            * np.exp(-((self._beta2 * energies / self.mean_E) ** self.g))
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


class Maxwellian(EEDF):
    def __init__(self, Te: float | None = None, mean_E: float | None = None):
        super().__init__(g=1.0, Te=Te, mean_E=mean_E)

    def pdf(self, energies: ArrayLike) -> np.ndarray:
        energies = np.asarray(energies, dtype=float)

        return (
            _maxwellian_beta1
            * np.sqrt(energies)
            * np.exp(-_maxwellian_beta2 * energies / self.mean_E)
            / self.mean_E**1.5
        )


class Druyvesteyn(EEDF):
    def __init__(self, Te: float | None = None, mean_E: float | None = None):
        super().__init__(g=2.0, Te=Te, mean_E=mean_E)

    def pdf(self, energies: ArrayLike) -> np.ndarray:
        energies = np.asarray(energies, dtype=float)

        return (
            _druyvesteyn_beta1
            * np.sqrt(energies)
            * np.exp(-((_druyvesteyn_beta2 * energies / self.mean_E) ** 2.0))
            / self.mean_E**1.5
        )
