import numpy as np


def interpolate_xs(energy: np.ndarray, xs_energy: np.ndarray, xs: np.ndarray):
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
