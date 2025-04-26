import numpy as np
from typing import List, Generator


class CoupledMapLattice:
    """An implementation of a coupled map lattice (CML) model."""

    def __init__(self, n: int, r: float, epsilion: float = 1) -> None:
        self.n = n
        self.r = r
        self.epsilion = epsilion
        self.state = np.random.normal(0, 1, (n, n))
        self.history = [self.state]
        self.time = 0

    @property
    def state(self) -> np.ndarray:
        """Returns the current state of the lattice."""
        return self._state

    @state.setter
    def state(self, value: np.ndarray) -> None:
        """Sets the state of the lattice.
        Args:
            value (np.ndarray): The new state of the lattice.
        """
        if not isinstance(value, np.ndarray):
            raise ValueError("State must be a numpy array.")
        if value.ndim != 2:
            raise ValueError("State must be a 2D array.")
        if value.dtype != np.float64:
            raise ValueError("State must be a float64 array.")

        if value.shape != (self.n, self.n):
            raise ValueError(f"State must be of shape ({self.n}, {self.n}).")
        self._state = value

    @property
    def history(self) -> List[np.ndarray]:
        """Returns the history of the lattice."""
        return self._history

    @history.setter
    def history(self, value: List[np.ndarray]) -> None:
        """Sets the history of the lattice.
        Args:
            value (List[np.ndarray]): The new history of the lattice.
        """
        if not isinstance(value, list):
            raise ValueError("History must be a list.")
        if not all(isinstance(x, np.ndarray) for x in value):
            raise ValueError("All elements in history must be numpy arrays.")

        self._history = value

    def state_function(self, x: np.ndarray) -> np.ndarray:
        """Applies a function to the state of the lattice.
        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output array after applying the function.
        """
        return self.r * x * (1 - x)

    def update(self) -> None:
        """Updates the state of the lattice.
        If `coupled` is True, the update is coupled.
        """
        if self.epsilion < 1:
            self._update_coupled()
        else:
            self._update_independent()
        self.history.append(self.state.tolist())
        self.time += 1

    def _update_coupled(self) -> None:
        """Updates the state of the lattice using a coupled map."""
        new_lattice = self.state.copy()
        for i in range(self.n):
            left_neighbor = self.state[(i - 1) % self.n]
            for j in range(self.n):
                # Apply the coupled map update
                new_lattice[i, j] = (
                    self.epsilion * self.state_function(self.state[i, j])
                    + (1 - self.epsilion) * self.state_function(left_neighbor[j])
                )

            # new_lattice = self.epsilion * self.state_function(self.state) + (
            #     1 - self.epsilion
            # ) * self.state_function(left_neighbor)
        self.state = new_lattice

    def _update_independent(self) -> None:
        """Updates the state of the lattice using an independent map."""
        self.state = self.state_function(self.state)

    def get_state(self) -> np.ndarray:
        """Returns the current state of the lattice."""
        return self.state.copy()

    def set_state(self, state: np.ndarray) -> None:
        """Sets the state of the lattice.
        Args:
            state (np.ndarray): The new state of the lattice.
        """
        self.state = state

    def get_history(self) -> List[np.ndarray]:
        """Returns the history of the lattice.

        Returns:
            List[np.ndarray]: The history of the lattice.
        """
        return self.history.copy()

    def reset(self) -> None:
        """Resets the lattice to its initial state."""
        self.state = np.random.normal(0, 1, (self.n, self.n))
        self.history = []
        self.time = 0

    def simulate(self, steps: int) -> Generator[np.ndarray, None, None]:
        """Simulates the lattice for a given number of steps.
        Args:
            steps (int): The number of steps to simulate.

        Yields:
            np.ndarray: The state of the lattice at each step.
        """
        for _ in range(steps):
            self.update()
            self.time += 1
            yield self.state.copy()
