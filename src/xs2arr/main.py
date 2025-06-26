from sys import exit

from xs2arr.model import Model
from xs2arr.io import get_args


def run() -> int:
    args = get_args()

    model = Model(args.input_file)

    return 0


if __name__ == "__main__":
    exit(run())
