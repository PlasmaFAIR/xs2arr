import numpy as np
from numpy.typing import ArrayLike


def interpolate_xs(
    energy: ArrayLike, xs_energy: ArrayLike, xs: ArrayLike
) -> np.ndarray:
    """
    Interpolate cross section data onto supplied energy grid using logarithmic interpolation. Returns the
    interpolated cross section values.

    Parameters
    ----------
    energy
        Energy points at which to interpolate the cross section.
    xs_energy
        Energy points of the original cross section data.
    xs
        Cross section values corresponding to xs_energy points.

    Raises
    ------
    ValueError
        If any values in xs are negative.

    Notes
    -----
    The function performs logarithmic interpolation by:
    1. Adding a (0,0) point to the data
    2. Converting cross sections to 10*log10 scale
    3. Performing linear interpolation
    4. Converting back to linear scale
    """
    energy = np.asarray(energy, dtype=float)
    xs_energy = np.asarray(xs_energy, dtype=float)
    xs = np.asarray(xs, dtype=float)

    if np.any(xs < 0.0):
        raise ValueError("All values in xs must be >= 0.0")

    # Add extra (0, 0) point to data.
    xs_energy = np.concatenate(([0.0], xs_energy))
    xs = np.concatenate(([0.0], xs))

    # We interpolate on the 10*log of the cross section.
    with np.errstate(divide="ignore"):
        xs_log = np.where(xs > 0.0, 10.0 * np.log10(xs), -np.inf)

    # Interpolate.
    xs_interp = np.interp(energy, xs_energy, xs_log)

    # And return the data with the inverse of the 10.0*log applied.
    return 10.0 ** (xs_interp / 10.0)
