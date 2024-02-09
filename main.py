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




x = np.zeros()