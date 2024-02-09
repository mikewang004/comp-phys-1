import numpy as np
import matplotlib.pyplot as plt

#Define constants

e_kb = 119.8 #K
sigma = 3.405 #A
h = 0.001 #timestep 
N = 10 #number of particles

#Note nabla U = 4 * epsilon (( -12 * sigma**12 * r**(-13)) - 6*sigma**6 * r**(-7))


def lennard_jones(epsilon, sigma, r):
    "Returns Lennard-Jones potential as given"
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)


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




def euler_position(x, v, h):
    "First order Euler approximation returns a position"
    return x + v * h

def euler_velocity(v, m, potential, h):
    "First order Euler approximation note potential requires a function"
    return v + 1/m * potential * h


x = np.zeros()