from collections.abc import Callable
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

_log10_e = np.log10(np.exp(1.0))


def arrhenius(T: ArrayLike, a: float, b: float, c: float) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    return a * T**b * np.exp(c / T)


def arrhenius_log(T: ArrayLike, log10_a: float, b: float, c: float) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    return log10_a + b * np.log10(T) + _log10_e * c / T


def validate_real(
    value: Real, name: str, condition: Callable[[float], bool] | None = None
) -> float:
    """
    Validate and convert a number to float, with optional positivity check.

    Parameters
    ----------
    value
        The value to validate and convert
    name
        Name of the parameter for error messages
    condition : optional
        If not None, evaluates the boundary condition with the validated Real.

    Raises
    ------
    TypeError
        If value is not a Real number (float or int)
    ValueError
        If condition is not met if provided
    """

    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a float or int, not {type(value).__name__}")

    value = float(value)

    if condition is not None and not condition(value):
        raise ValueError(f"{name} does not satisfy the condition")

    return value
