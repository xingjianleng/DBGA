import os
import pytest

@pytest.fixture
def data_dir():
    # returns the path to the data directory
    return os.path.join(os.path.dirname(__file__), 'data')