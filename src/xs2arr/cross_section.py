from lxcat_data_parser import CrossSectionSet


def read_cross_sections_from_file(filename: str):
    return CrossSectionSet(filename)
