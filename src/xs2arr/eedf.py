from abc import ABC, abstractmethod

from numpy import exp, ndarray

from xs2arr.constants import (
    druyvesteyn_beta1,
    druyvesteyn_beta2,
    maxwellian_beta1,
    maxwellian_beta2,
)


class EEDF(ABC):
    def __init__(self, Te: float | None = None, mean_E: float | None = None):
        if Te is None and mean_E is None:
            raise ValueError("Either Te or mean_E must be provided")
        if Te is not None and mean_E is not None:
            raise ValueError("Only one of Te or mean_E must be provided")
        if mean_E is not None:
            Te = 2.0 * mean_E / 3.0
        self._Te = Te
        self._mean_E = 3.0 * Te / 2.0

    @abstractmethod
    def pdf(self, energies: ndarray) -> ndarray:
        pass

    @property
    def Te(self) -> float:
        return self._Te

    @property
    def mean_E(self) -> float:
        return self._mean_E


class Maxwellian(EEDF):
    def pdf(self, energies: ndarray) -> ndarray:
        return (
            maxwellian_beta1
            * energies**0.5
            * exp(-maxwellian_beta2 * energies / self.mean_E)
            / self.mean_E**1.5
        )


class Druyvesteyn(EEDF):
    def pdf(self, energies: ndarray) -> ndarray:
        return (
            2.0
            * druyvesteyn_beta1
            * energies**0.5
            * exp(-druyvesteyn_beta2 * (energies / self.mean_E) ** 2.0)
            / self.mean_E**1.5
        )
