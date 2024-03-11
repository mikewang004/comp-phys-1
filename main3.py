import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants as spc
#Define constants


test_seed = 100

#Note nabla U = 4 * epsilon (( -12 * sigma**12 * r**(-13)) - 6*sigma**6 * r**(-7))


def lennard_jones_natural(dists_nat):
    "Returns Lennard-Jones potential as given. r_natural is the distance in rescaled (natural) units"
    return 4 * epsilon * ((dists_nat)**-12 - (dists_nat)**-6)

    

# def particle_connecting_vectors(particle_positions):


def forces(particle_distances_arr, particle_positions):
    dimensions = np.shape(particle_positions)[1]
    unit_diff_vectors = np.zeros((np.shape(particle_positions), dimensions))

    for dim in dimensions: 
        dists_along_axis = sp.spatial.distance.cdist(particle_positions[dim])
        unit_diff_vectors[:,:,dim] = dists_along_axis
    return 4*(-12 * particle_distances_arr**-12.0 + 6 * particle_distances_arr**-7.0)

def periodic_bcs(positions, box_length):
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
    
def euler_velocity(v, h, forces ):
    "First order Euler approximation note potential requires a function"
    return v + forces * h


def time_step(positions, velocities, h, L):

    #First look for smallest distances within each array to apply potential to then update whole diagram
    particle_distances = sp.spatial.distance.cdist(x_0, x_0)
    particle_forces = forces(particle_distances)

    positions_new = euler_position(positions, velocities, h)
    velocities_new = euler_velocity(velocities, h, forces)

    positions_new = periodic_bcs(positions_new, L) # apply bcs


    return positions_new, velocities_new 

def time_loop(initial_positions, initial_velocities, h, max_time, L):
    N_particles = np.shape(initial_positions)[0]
    N_timesteps = int(max_time/h)
    N_dimensions = np.shape(initial_positions)[1]

    particle_positions = initial_positions
    particle_velocities = initial_velocities
    results_positions = np.zeros((N_timesteps, N_particles, N_dimensions))  
    results_velocities = np.zeros((N_timesteps, N_particles, N_dimensions)) 
    results_positions[0, :, :] = initial_positions
    results_velocities[0, :, :] = initial_velocities

    for i in range(0, max_time):
        particle_positions, particle_velocities = time_step(particle_positions, particle_velocities, h, L) 
        results_positions[i,:,:] = particle_positions
        results_velocities[i,:,:] = particle_velocities

    return results_positions, results_velocities



h=0.1
N = 10
dim=2
max_time =2
L = 2

rng = np.random.default_rng()
x_0 = rng.uniform(low = -10, high = 10, size = (N, dim))
v_0 = rng.uniform(low = -3, high = 3, size = (N, dim))

results = time_loop(x_0, v_0, h, max_time, L)




#time_step(x_0, v_0, h)
x, v, x_all = time_loop(x_0, v_0, h, max_time)