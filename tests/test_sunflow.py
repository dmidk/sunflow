"""Basic tests for the sunflow package."""

import sunflow


def test_version():
    """Ensure the package exposes a version string."""
    assert isinstance(sunflow.__version__, str)
    assert len(sunflow.__version__) > 0
