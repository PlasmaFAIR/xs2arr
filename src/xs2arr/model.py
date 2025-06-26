from xs2arr.cross_section import read_cross_sections_from_file


class Model:
    def __init__(self, input_file: str = None):
        if input_file is None:
            raise ValueError('No input file provided')

        self.cross_sections = read_cross_sections_from_file(input_file)
