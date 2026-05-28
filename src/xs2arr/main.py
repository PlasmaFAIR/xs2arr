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
    parser.add_argument(
        "-a",
        "--append",
        action="store_true",
        default=False,
        help="Append to output file",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Verbose output"
    )

    args = parser.parse_args()

    model = Model(lxcat_file=args.input_file)

    model.fit()

    model.write_results(output_file=args.output_file, append_to_file=args.append)

    return 0


if __name__ == "__main__":
    exit(run())
