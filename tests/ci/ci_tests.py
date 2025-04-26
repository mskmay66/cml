from cml import CoupledMapLattice
import numpy as np

def test_update():
    """Test the update method of the CoupledMapLattice class."""
    # Create a CoupledMapLattice instance
    lattice = CoupledMapLattice(10, r=1)

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
    assert lattice.state.flatten().tolist() == (lattice.r * initial_state * (1 - initial_state)).flatten().tolist(), (
        "State update formula is incorrect."
    )

    coupled_lattice = CoupledMapLattice(10, r=0.5, epsilion=0.5)
    coupled_lattice.update()
    assert lattice.state.shape == (10, 10), (
        "State should have the same shape as the lattice."
    )
    assert lattice.state.dtype == np.float64, "State should be of type float64."
    assert lattice.state.ndim == 2, "State should be a 2D array."


def test_simulate():
    """Test the simulate method of the CoupledMapLattice class."""
    # Create a CoupledMapLattice instance
    lattice = CoupledMapLattice(10, r=0.5)
    initial_state = lattice.state.copy()

    # Simulate for 5 time steps
    simulator = lattice.simulate(5)

    assert simulator is not None, "Simulator should not be None."
    assert sum(1 for _ in simulator) == 5, (
        "Simulator should yield 5 time steps."
    )

    assert len(lattice.history) == 6, (
        "History should contain 6 elements after simulating 5 steps."
    )

    assert np.array_equal(lattice.history[0], initial_state), (
        "First element in history should be the initial state."
    )