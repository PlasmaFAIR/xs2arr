from lxcat_data_parser import CrossSectionSet


def parse_lxcat_data(filename: str) -> CrossSectionSet:
    """Parse LXCat data file and return a CrossSectionSet object.

    Parameters
    ----------
    filename : str
        Path to the LXCat data file to be parsed.

    Raises
    ------
    TypeError
        If filename is not a string.
    ValueError
        If filename is an empty string.
    """

    if not isinstance(filename, str):
        raise TypeError("filename must be of type str")
    if not filename:
        raise ValueError("filename cannot be an empty string")

    return CrossSectionSet(filename)
