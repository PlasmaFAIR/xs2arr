from xs2arr import Model


EXAMPLE_LXCAT_FILE = "tests/example_lxcat.txt"


def test_model():
    """Test the robustness of creating the model class from example LXCat data."""

    Model(EXAMPLE_LXCAT_FILE)
