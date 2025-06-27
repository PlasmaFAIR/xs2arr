from xs2arr import Model, EXAMPLE_LXCAT_FILE


def test_model():
    """Test the robustness of creating the model class from example LXCat data."""

    Model(EXAMPLE_LXCAT_FILE)
