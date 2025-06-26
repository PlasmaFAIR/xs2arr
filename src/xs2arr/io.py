from argparse import ArgumentParser
from lxcat_data_parser import CrossSectionSet


def parse_lxcat_data(filename):
    return CrossSectionSet(filename)


def get_args():
    parser = ArgumentParser(
        prog="Cross section to Arrhenius",
        description="Calculates Arrhenius coefficients from a provided cross section",
    )

    parser.add_argument(
        "input_file", type=str, default="input.txt", nargs="?", help="Input LXCat file"
    )
    parser.add_argument(
        "output_file", type=str, default="output.txt", nargs="?", help="Output file"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    return args
