import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants as spc
#Define constants


test_seed = 100

#Note nabla U = 4 * epsilon (( -12 * sigma**12 * r**(-13)) - 6*sigma**6 * r**(-7))


def lennard_jones_natural(dists_nat):
    # FIX THIS
    epsilon = 1
    "Returns Lennard-Jones potential as given. r_natural is the distance in rescaled (natural) units"
    return 4 * epsilon * ((dists_nat)**-12 - (dists_nat)**-6)

    

# def particle_connecting_vectors(particle_positions):


def forces(particle_positions, particle_distances_arr):
    """return net force array for all particles

    Args:
        particle_positions (arr): positions in N-d for the particles
        particle_distances_arr (arr): N x N array
    """
    # still not fully complete
    dimensions = np.shape(particle_positions)[1] # number of columns in the positions array corresponds to the dimension
    N_particles = np.shape(particle_positions)[0]
    diff_matrix = np.zeros((N_particles,N_particles, dimensions)) # the diff_matrix contains all possible combinations of vectors between a set of two particles

    # calculate diff matrix across all dimensions
    for dim in range(0, dimensions): 
        dists_along_axis  = np.tile(particle_positions[:,0], (N_particles, 1)) - np.tile(particle_positions[:,0], (N_particles, 1)).T
        diff_matrix[:,:,dim] = dists_along_axis # arr of size N x N
    
    # make a matrix to store the inverted norm of all diff vectors so we can normalize them
    diff_matrix_inv_norm  = np.zeros((N_particles, N_particles))
    diff_matrix_inv_norm[:,:] = 1/np.sqrt((np.sum(diff_matrix[:,:]**2))) # sum the squares of elements across all dimensions

    # do the normalization    
    norm_diff_matrix = diff_matrix.copy()
    norm_diff_matrix[:,:,:] = norm_diff_matrix[:,:,:] * diff_matrix_inv_norm

    # forces = 4*(-12 * particle_distances_arr**-12.0 + 6 * particle_distances_arr**-7.0) 
    return 0;

def zero_forces(particle_positions, particle_distances_arr):
    net_force = 0.0 * particle_positions
    return net_force
    

def periodic_bcs(positions, box_length):
    """Apply periodic boundry conditions by subtracting L in the  'violating' components

    Args:
        posititions (array): position array
        velocities (array): velocities array
    """
    
    positions[positions > box_length/2] = positions[positions > box_length/2] - box_length
    positions[positions < -box_length/2] = positions[positions < -box_length/2] + box_length

    return positions

    

def euler_position(x, v, h):
    "First order Euler approximation returns a position"
    return x + v * h
    
def euler_velocity(v, h, net_forces ):
    """perform euler forward on velocity

    Args:
        v (arr): N x 2 velocity array
        h (float): step size
        forces (arr): N x 2 net forces

    Returns:
        _type_: _description_
    """
    "First order Euler approximation note potential requires a function"
    return v + net_forces* h


def time_step(positions, velocities, h, L):

    #First look for smallest distances within each array to apply potential to then update whole diagram
    particle_distances = sp.spatial.distance.cdist(positions, positions)
    particle_forces = zero_forces(positions, particle_distances)
    print(f'{np.shape(positions)=}')
    print(f'{np.shape(particle_forces)=}')

    positions_new = euler_position(positions, velocities, h)
    velocities_new = euler_velocity(velocities, h, particle_forces)

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

    for i in range(0, N_timesteps):
        particle_positions, particle_velocities = time_step(particle_positions, particle_velocities, h, L) 
        results_positions[i,:,:] = particle_positions
        results_velocities[i,:,:] = particle_velocities

    return results_positions, results_velocities



h=0.005
N = 10
dim=2
max_time =2
L = 2

rng = np.random.default_rng()
x_0 = rng.uniform(low = -L/2, high = L/2, size = (N, dim))
v_0 = rng.uniform(low = -1, high = 1, size = (N, dim))

loop_results_x, loop_results_v = time_loop(x_0, v_0, h, max_time, L)

print(np.shape(loop_results_x))
plt.plot(loop_results_x[:,:,0], loop_results_x[:,:,1], marker='x')
plt.show()




#time_step(x_0, v_0, h)
# x, v, x_all = time_loop(x_0, v_0, h, max_time)
