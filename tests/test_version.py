"""Test package metadata."""

import conformal


def test_version_exists() -> None:
    """Test that __version__ is defined."""
    assert hasattr(conformal, "__version__")
    assert isinstance(conformal.__version__, str)
    assert len(conformal.__version__) > 0
