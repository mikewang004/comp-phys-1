import numpy as np
import matplotlib.pyplot as plt

#Define constants

e_kb = 119.8 #K
sigma = 3.405 #A
h = 0.001 #timestep 
N = 10 #number of particles

#Note nabla U = 4 * epsilon (( -12 * sigma**12 * r**(-13)) - 6*sigma**6 * r**(-7))


def lennard_jones(epsilon, sigma, r):
    """Calcultate Lennard Jones potential

    Args:
        epsilon (float): 
        sigma (float): 
        r (float): distance between two particles

    Returns:
        potential (float): calculated potential
    """
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)


def periodic_bcs(positions, velocities, box_length):
    """Apply periodic boundry conditions by subtracting L ONCE in the  'violating' components

    Args:
        posititions (array): position array
        velocities (array): velocities array
    """
    
    positions[positions > box_length] = positions[positions > box_length] - box_length
    positions[positions < 0] = positions[positions < 0] + box_length

    return positions





x = np.zeros()