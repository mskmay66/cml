from __future__ import annotations

from .cml import CoupledMapLattice


class KanekoLattice(CoupledMapLattice):
    """An implementation of the Kaneko map."""

    def __init__(self, n: int, r: float, epsilion: float = 1) -> None:
        super().__init__(n, r, epsilion)

    def __repr__(self):
        return f"KenekoLattice(n={self.n}, r={self.r}, epsilion={self.epsilion})"

    def update(self):
        """Updates the state of the lattice using the Kaneko map."""
        state = self.state.copy()
        for i in range(self.n):
            left_neighbor = state[(i - 1) % self.n]
            right_neighbor = state[(i + 1) % self.n]
            for j in range(self.n):
                # Apply the Kaneko map update
                state[i, j] = self.epsilion * self.state_function(state[i, j]) + (
                    self.epsilion / 2
                ) * (
                    self.state_function(
                        left_neighbor[j] +
                        self.state_function(right_neighbor[j]),
                    )
                )
        self.state = state
        self.history.append(self.state.tolist())
        self.time += 1
