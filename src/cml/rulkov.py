from .cml import CoupledMapLattice
import numpy as np


class RulkovLattice(CoupledMapLattice):
    """An implementation of the Rulkov map."""

    def __init__(self, n: int, r: float, mu: float, sigma: float, epsilion: float = 1) -> None:
        super().__init__(n, r, epsilion)
        self.state = np.random.normal(0, 1, (2, n, n))
        self.mu = mu
        self.sigma = sigma
    

    def state_function(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Applies the Rulkov map update function to the state of the lattice.

        Args:
            x (np.ndarray): The input array representing the first state.
            y (np.ndarray): The input array representing the second state.

        Returns:
            np.ndarray: The updated state.
        """
        x_next = (self.r / (1 + x**2)) + y
        y_next = y - self.mu * (x_next - self.sigma)
        self.state[0] = x_next
        self.state[1] = y_next
    

    def _update_coupled(self) -> None:
        """Updates the state of the lattice using the Rulkov map."""
        state = self.state.copy()
        for i in range(self.n):
            left_neighbor = state[:, (i - 1) % self.n]
            right_neighbor = state[:, (i + 1) % self.n]
            for j in range(self.n):
                # Apply the Rulkov map update
                state[:, i, j] = (
                    self.epsilion * self.state_function(state[0, i, j], state[1, i, j])
                    + (self.epsilion / 2) * (
                        self.state_function(left_neighbor[0] + self.state_function(left_neighbor[1]), 
                                            left_neighbor[1] + self.state_function(right_neighbor[1]))
                    )
                )
        self.state = state
    

    def _update_independent(self) -> None:
        """Updates the state of the lattice using an independent map."""
        self.state = self.state_function(self.state[0], self.state[1])
    

    def update(self):
        if self.epsilion < 1:
            self._update_coupled()
        else:
            self._update_independent()
        self.history.append(self.state.tolist())
        self.time += 1
