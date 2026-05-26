"""Test that project setup is correct."""


def test_package_importable():
    """Verify tsxr2 package can be imported."""
    import tsxr2

    assert tsxr2.__version__ is not None
