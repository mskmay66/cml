from cml import CoupledMapLattice
import numpy as np


def test_update():
    """Test the update method of the CoupledMapLattice class."""
    # Create a CoupledMapLattice instance
    lattice = CoupledMapLattice(10)

    initial_state = lattice.state.copy()
    lattice.update()
    assert not np.array_equal(initial_state, lattice.state), (
        "State should change after update."
    )
    assert len(lattice.history) == 2, (
        "History should contain one element after first update."
    )
    assert np.array_equal(lattice.history[0], initial_state), (
        "History should contain the initial state."
    )
    assert lattice.time == 1, "Time should increment after update."
    assert lattice.state.shape == (10, 10), (
        "State should have the same shape as the lattice."
    )
    assert lattice.state.dtype == np.float64, "State should be of type float64."
    assert lattice.state.ndim == 2, "State should be a 2D array."
    assert lattice.state == lattice.r * lattice.state * (1 - lattice.state), (
        "State update formula is incorrect."
    )

    coupled_lattice = CoupledMapLattice(10, coupled=True)
    coupled_lattice.update()
    assert lattice.state.shape == (10, 10), (
        "State should have the same shape as the lattice."
    )
    assert lattice.state.dtype == np.float64, "State should be of type float64."
    assert lattice.state.ndim == 2, "State should be a 2D array."


def test_simulate():
    """Test the simulate method of the CoupledMapLattice class."""
    # Create a CoupledMapLattice instance
    lattice = CoupledMapLattice(10)

    # Simulate for 5 time steps
    lattice.simulate(5)

    assert len(lattice.history) == 6, (
        "History should contain 6 elements after simulating 5 steps."
    )
    assert lattice.time == 5, "Time should be 5 after simulating 5 steps."
    assert np.array_equal(lattice.history[0], lattice.state), (
        "First element in history should be the initial state."
    )
