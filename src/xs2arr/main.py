from argparse import ArgumentParser
from sys import exit

from xs2arr.model import Model


def run() -> int:
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

    model = Model(args.input_file)  # noqa: F841

    return 0


if __name__ == "__main__":
    exit(run())
