import geom_analysis as ga
import pytest

def test_calc_dist():
    """Test the bond length calculation."""
    coord1 = [0.0, 0.0, 0.0]
    coord2 = [1.0, 0.0, 0.0]
    expected = 1.0
    calculated = ga.calc_dist(coord1, coord2)
    assert abs(calculated - expected) < 1e-6, f"Expected {expected}, got {calculated}"
def test_bond_check():
    """Test the bond check function."""
    assert ga.bond_check(1.0) is True, "Expected bond length to be valid"
    assert ga.bond_check(1.5) is True, "Expected bond length to be valid"
    assert ga.bond_check(0) is False, "Expected bond length to be invalid"
    assert ga.bond_check(2.0) is False, "Expected bond length to be invalid"
    assert ga.bond_check(0.5, min_length=0.1, max_length=1.5) is True, "Expected bond length to be valid with custom limits"
    assert ga.bond_check(0.05, min_length=0.1, max_length=1.5) is False, "Expected bond length to be invalid with custom limits"

def test_open_xyz_error():
    """Test the open_xyz function flagging a non .xyz file."""
    fname = 'test.txt'
    with pytest.raises(ValueError):
        ga.open_xyz(fname)

def test_bond_check_negative():
    distance = -1
    """Test the bond_check function with a negative distance."""
    with pytest.raises(ValueError):
        calculated = ga.bond_check(distance)