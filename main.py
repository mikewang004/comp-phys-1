import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants as spc
#Define constants

e_kb = 119.8 #K
epsilon = e_kb * sp.constants.Boltzmann
sigma = 3.405 #A
h = 0.001 #timestep 
N = 10 #number of particles
dim = 2 #number of spacelike dimensions
box_length = 10

#Note nabla U = 4 * epsilon (( -12 * sigma**12 * r**(-13)) - 6*sigma**6 * r**(-7))


def lennard_jones(r, epsilon, sigma):
    "Returns Lennard-Jones potential as given. r is distance between atoms."
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

def nabla_lennard_jones(r, epsilon, sigma):
    "Derivative of the Lennard-Jones potential"
    return -3 * epsilon * ((sigma)**12 / r**(-13)) + 2/3 * epsilon * ((sigma)**6 / r**(-7))

def potential(x1, x2, potential_function, epsilon, sigma):
    """Assuming distance-only potential given by dU/dr * x_vector / r. Note potential here assumed to be 
    dU/dr i.e. nabla is assumed to be already applied to the potential."""
    r = np.linalg.norm(x1 - x2)
    return potential_function(r, epsilon, sigma)

def periodic_bcs(positions, velocities, box_length):
    """Apply periodic boundry conditions by subtracting L ONCE in the  'violating' components

    Args:
        posititions (array): position array
        velocities (array): velocities array
    """
    
    positions[positions > box_length] = positions[positions > box_length] - box_length
    positions[positions < 0] = positions[positions < 0] + box_length

    return positions

def euler_position(x, v, h):
    "First order Euler approximation returns a position"
    return x + v * h

def euler_velocity(v, m, potential, h):
    "First order Euler approximation note potential requires a function"
    return v + 1/m * potential * h




#Define starting conditions
rng = np.random.default_rng()
x_0 = rng.uniform(low = -box_length, high = box_length, size = (N, dim))
v_0 = rng.uniform(low = -box_length, high = box_length, size = (N, dim))

print(v_0)