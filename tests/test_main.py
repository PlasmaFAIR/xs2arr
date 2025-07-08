from xs2arr import Model

EXAMPLE_LXCAT_FILE = "tests/example_lxcat.txt"


def test_model(snapshot):
    """Test the robustness of the model class from an example LXCat data."""

    model = Model(EXAMPLE_LXCAT_FILE)
    results = model.arrhenius()

    assert results == snapshot
