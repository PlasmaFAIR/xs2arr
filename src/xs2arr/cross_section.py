from numpy import concatenate, interp, log10, ndarray


def interpolate_xs(energy: ndarray, xs_energy: ndarray, xs: ndarray):
    # Add extra (0, 0) point to data.
    xs_energy = concatenate(([0.0], xs_energy))
    xs = concatenate(([0.0], xs))

    # We interpolate on the 10*log of the cross section.
    xs_log = 10.0 * log10(xs)

    # Interpolate.
    xs_interp = interp(energy, xs_energy, xs_log)

    # And return the data with the inverse of the 10.0*log applied.
    return 10.0 ** (xs_interp / 10.0)
