from .kaneko import KenekoLattice
from .rulkov import RulkovLattice
import numpy as np

def test_kaneko():
    """Test the kaneko latttice."""
    kaneko_decoupled = KenekoLattice(10, 0.5, 1)
    initial_state = kaneko_decoupled.state.copy()
    kaneko_decoupled.update()
    assert not np.array_equal(initial_state, kaneko_decoupled.state), (
        "State should change after update."
    )
    assert len(kaneko_decoupled.history) == 2, (
        "History should contain one element after first update."
    )
    assert np.array_equal(kaneko_decoupled.history[0], initial_state), (
        "History should contain the initial state."
    )
    assert kaneko_decoupled.time == 1, "Time should increment after update."
    assert kaneko_decoupled.state.shape == (10, 10), (
        "State should have the same shape as the lattice."
    )
    assert kaneko_decoupled.state.dtype == np.float64, "State should be of type float64."
    assert kaneko_decoupled.state.ndim == 2, "State should be a 2D array."
    assert kaneko_decoupled.state.flatten().tolist() == (kaneko_decoupled.r * initial_state * (1 - initial_state)).flatten().tolist(), (
        "State update formula is incorrect."
    )

    assert kaneko_decoupled.state.shape == (10, 10), (
        "State should have the same shape as the lattice."
    )
    assert kaneko_decoupled.state.dtype == np.float64, "State should be of type float64."
    assert kaneko_decoupled.state.ndim == 2, "State should be a 2D array." 

    kaneko_coupled = KenekoLattice(10, r=0.5, epsilion=0.5)
    initial_state = kaneko_coupled.state.copy()
    kaneko_coupled.update()
    assert not np.array_equal(initial_state, kaneko_coupled.state), (
        "State should change after update."
    )
    assert len(kaneko_coupled.history) == 2, (
        "History should contain one element after first update."
    )
    assert np.array_equal(kaneko_coupled.history[0], initial_state), (
        "History should contain the initial state."
    )
    assert kaneko_coupled.time == 1, "Time should increment after update."
    assert kaneko_coupled.state.shape == (10, 10), (
        "State should have the same shape as the lattice."
    )
    assert kaneko_coupled.state.dtype == np.float64, "State should be of type float64."
    assert kaneko_coupled.state.ndim == 2, "State should be a 2D array."


def test_rulkov():
    """Test the rulkov latttice."""
    rulkov_decoupled = RulkovLattice(10, 0.5, 1, 1)
    initial_state = rulkov_decoupled.state.copy()
    rulkov_decoupled.update()
    assert not np.array_equal(initial_state, rulkov_decoupled.state), (
        "State should change after update."
    )
    assert len(rulkov_decoupled.history) == 2, (
        "History should contain one element after first update."
    )
    assert np.array_equal(rulkov_decoupled.history[0], initial_state), (
        "History should contain the initial state."
    )
    assert rulkov_decoupled.time == 1, "Time should increment after update."
    assert rulkov_decoupled.state.shape == (2, 10, 10), (
        "State should have the same shape as the lattice."
    )
    assert rulkov_decoupled.state.dtype == np.float64, "State should be of type float64."
    assert rulkov_decoupled.state.ndim == 3, "State should be a 3D array."
    assert rulkov_decoupled.state.flatten().tolist() == (rulkov_decoupled.r * initial_state * (1 - initial_state)).flatten().tolist(), (
        "State update formula is incorrect."
    )

    rulkov_coupled = RulkovLattice(10, 0.5, 1, 0.5)
    initial_state = rulkov_coupled.state.copy()
    rulkov_decoupled.update()
    assert not np.array_equal(initial_state, rulkov_coupled.state), (
        "State should change after update."
    )
    assert len(rulkov_coupled.history) == 2, (
        "History should contain one element after first update."
    )
    assert np.array_equal(rulkov_coupled.history[0], initial_state), (
        "History should contain the initial state."
    )
    assert rulkov_coupled.time == 1, "Time should increment after update."
    assert rulkov_coupled.state.shape == (2, 10, 10), (
        "State should have the same shape as the lattice."
    )
    assert rulkov_coupled.state.dtype == np.float64, "State should be of type float64."
    assert rulkov_coupled.state.ndim == 3, "State should be a 3D array."
