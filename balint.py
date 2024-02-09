import numpy as np
import matplotlib.pyplot as plt


def periodic_bcs(posititions, velocities):
    """Apply periodic boundry conditions 

    Args:
        posititions (array): position array
        velocities (array): velocities array
    """


def do_timestep(positions, velocities):
    positions = positions + velocities * dt
    return positions


N = 20
dt = 0.1
positions = np.random.random((N, 2))
velocities = np.random.random((N, 2))


for i in range(0,10):
    positions = do_timestep(positions)
    box_length = 1
    positions[positions > box_length] = positions[positions > box_length] - box_length
    positions[positions < 0] = positions[positions < 0] + box_length
    plt.plot(positions[])
