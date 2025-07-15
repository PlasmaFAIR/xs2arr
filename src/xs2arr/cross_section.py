from numpy import concatenate, errstate, inf, interp, log10, ndarray, where


def interpolate_xs(energy: ndarray, xs_energy: ndarray, xs: ndarray):
    if any(xs < 0.0):
        raise ValueError("All values in xs must be >= 0.0")

    # Add extra (0, 0) point to data.
    xs_energy = concatenate(([0.0], xs_energy))
    xs = concatenate(([0.0], xs))

    # We interpolate on the 10*log of the cross section.
    with errstate(divide="ignore"):
        xs_log = where(xs > 0.0, 10.0 * log10(xs), -inf)

    # Interpolate.
    xs_interp = interp(energy, xs_energy, xs_log)

    # And return the data with the inverse of the 10.0*log applied.
    return 10.0 ** (xs_interp / 10.0)
