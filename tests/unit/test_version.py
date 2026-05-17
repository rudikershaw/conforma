"""Test package metadata."""

import conforma


def test_version_exists() -> None:
    """Test that __version__ is defined."""
    assert hasattr(conforma, "__version__")
    assert isinstance(conforma.__version__, str)
    assert len(conforma.__version__) > 0
