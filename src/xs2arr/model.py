from xs2arr.io import parse_lxcat_data


class Model:
    def __init__(self, input_file: str = None):
        if input_file is None:
            raise ValueError("No input file provided")

        self.data = parse_lxcat_data(input_file)
