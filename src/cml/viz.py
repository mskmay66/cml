import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from matplotlib import animation
from .cml import CoupledMapLattice
import os
from matplotlib.animation import PillowWriter
from datetime import datetime


class Visualization:
    """A class for visualizing the state of a Coupled Map Lattice (CML)."""

    def __init__(self) -> None:
        pass
    

    def __call__(self, lattice: CoupledMapLattice) -> None:
        """Update the visualization with the new lattice state.

        Args:
            lattice (CoupledMapLattice): The updated lattice.
        """
        assert isinstance(lattice, CoupledMapLattice), "lattice must be an instance of CoupledMapLattice."
        assert len(lattice.history) > 1, "History must contain at least two elements."
        self.lattice = lattice
        self.fig, self.ax = plt.subplots()
        self.animate(frames=len(self.lattice.history))
    

    def generate_filename(self) -> str:
        """Generate a filename for the animation based on the current date and time.

        Returns:
            str: The generated filename.
        """
        now = datetime.now()
        return now.strftime("lattice_animation_%Y%m%d_%H%M%S.gif")
    

    def init_animation(self) -> None:
        """Initialize the animation."""
        self.ax.clear()
        self.im = self.ax.imshow(
            self.lattice.state,
            cmap="plasma",
            interpolation="nearest",
            animated=True,
        )
        return self.im,


    def update(self, i: int) -> None:
        """Update the visualization for the given frame.

        Args:
            i (int): The current frame number.
        """
        self.im.set_array(np.nan_to_num(self.lattice.history[i]))

        self.ax.set_title(f"Time: {i}")
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.set_xticks([])
        return self.im,
    

    def animate(self, frames: Optional[int] = None) -> None:
        """Animate the visualization.

        Args:
            frames (Optional[int]): The number of frames to animate. If None, use the length of the history.
        """
        self.ax.clear()
        if frames is None:
            frames = len(self.lattice.history)

        ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=frames,
            init_func=self.init_animation,
            interval=50,
            blit=True,
            repeat_delay=1000
        )
        os.makedirs("map_animations", exist_ok=True)
        filename = self.generate_filename()
        ani.save(os.path.join("map_animations", filename), writer=PillowWriter(fps=30))

        plt.show()
