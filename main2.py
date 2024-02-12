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
m = 6.6 * 10**-26
natural_constant =  (m *sigma**2) / epsilon
v_start = sigma / np.sqrt(natural_constant)

#Note nabla U = 4 * epsilon (( -12 * sigma**12 * r**(-13)) - 6*sigma**6 * r**(-7))


def lennard_jones_natural(r, epsilon, sigma):
    "Returns Lennard-Jones potential as given. r is distance between atoms."
    r_sigma = r/sigma
    return 4 * epsilon * ((r_sigma)**-12 - (r_sigma)**-6)

def nabla_lennard_jones_natural(r_natural):
    return 4*(-12 * r_natural**-13 + 6 * r_natural**-7)


def potential(x1, x2, potential_function, epsilon, sigma):
    """Assuming distance-only potential given by dU/dr * x_vector / r. Note potential here assumed to be 
    dU/dr i.e. nabla is assumed to be already applied to the potential."""
    r = np.linalg.norm(x1 - x2)
    return potential_function(r, epsilon, sigma)

def potential_natural(x1, x2, potential_function):
    r_natural = np.linalg.norm(x1 - x2)
    return potential_function(r_natural)

def periodic_bcs(positions, velocities, box_length):
    """Apply periodic boundry conditions by subtracting L in the  'violating' components

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
    return v + potential * h


def time_loop(x, v, h, max_time = 100, potential=potential_natural):
    #First look for smallest distances within each array to apply potential to then update whole diagram



#Define starting conditions
rng = np.random.default_rng()
x_0 = rng.uniform(low = -box_length, high = box_length, size = (N, dim))
v_0 = rng.uniform(low = -3, high = 3, size = (N, dim))

print(v_0)