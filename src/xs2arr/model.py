from xs2arr.io import parse_lxcat_data


class Model:
    def __init__(self, lxcat_file: str | None = None):
        if lxcat_file is None:
            raise ValueError("No lxcat file provided")

        self.cross_section_set = parse_lxcat_data(lxcat_file)
