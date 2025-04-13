def assert_almost_equal(value1, value2, tolerance=1e-6):
    """
    Assert that two floating-point values are almost equal within a given tolerance.
    """
    assert abs(value1 - value2) <= tolerance, f"{value1} and {value2} differ by more than {tolerance}"
