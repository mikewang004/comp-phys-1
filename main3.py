import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants as spc

# Define constants


test_seed = 100

# Note nabla U = 4 * epsilon (( -12 * sigma**12 * r**(-13)) - 6*sigma**6 * r**(-7))


def lennard_jones_natural(dists_nat):
    epsilon = 1
    "Returns Lennard-Jones potential as given. r_natural is the distance in rescaled (natural) units"
    return 4 * epsilon * ((dists_nat) ** -12 - (dists_nat) ** -6)


def forces(particle_positions, particle_distances_arr):
    """return net force array for all particles

    Args:
        particle_positions (arr): positions in N-d for the particles
        particle_distances_arr (arr): N x N array
    """
    # number of columns in the positions array corresponds to the dimension
    dimensions = np.shape(particle_positions)[1]
    N_particles = np.shape(particle_positions)[0]

    # calculate diff matrix across all dimensions
    diff_matrix = calc_diff_matrix(particle_positions, dimensions, N_particles)
    norm_diff_matrix = normalize_diff_matrix(dimensions, N_particles, diff_matrix)

    # assuming particle_distances_arr has identical ordering but probably could be more general
    forces_magnitudes = np.zeros((np.shape(particle_distances_arr)))
    forces_magnitudes[:, :] = 4 * (
        -12 * particle_distances_arr[:, :] ** -13.0
        + 6 * particle_distances_arr[:, :] ** -7.0
    )
    forces_magnitudes[np.isnan(forces_magnitudes)] = 0
    net_force = np.zeros((N_particles, dimensions))
    repeated_force_magnitudes = np.repeat(
        forces_magnitudes[np.newaxis, :, :], dimensions, axis=0
    )
    # sum over other particles
    net_force = np.sum(repeated_force_magnitudes * norm_diff_matrix, axis=1).T

    return net_force


def normalize_diff_matrix(dimensions, N_particles, diff_matrix):
    """Normalize diff matrix to unit vectors

    Args:
        dimensions (int): number of dimensions
        N_particles (int): number of particles
        diff_matrix (arr): the diff matrix to normalize

    Returns:
        arr: normalized diff matrix
    """
    # make a matrix to store the inverted norm of all diff vectors so we can normalize them
    diff_matrix_inv_norm = np.zeros((N_particles, N_particles))
    # sum the squares of elements across all dimensions
    diff_matrix_inv_norm[:, :] = 1 / np.sqrt((np.sum(diff_matrix[:, :] ** 2, axis=0)))

    # do the normalization
    norm_diff_matrix = diff_matrix.copy()
    norm_diff_matrix[:, :, :] = norm_diff_matrix[:, :, :] * (
        np.repeat(diff_matrix_inv_norm[np.newaxis, :, :], dimensions, axis=0)
    )
    norm_diff_matrix[np.isnan(norm_diff_matrix)] = 0
    return norm_diff_matrix


def calc_diff_matrix(particle_positions, dimensions, N_particles):
    """
    the diff_matrix contains all possible combinations of vectors between a set of two particles

    Args:
        particle_positions (arr): (N x dim)
        dimensions (int): number of dimensions
        N_particles (int): number of particles

    Returns:
        _type_: _description_
    """
    diff_matrix = np.zeros((dimensions, N_particles, N_particles))
    for dim in range(0, dimensions):
        axis_positions = particle_positions[:, dim]
        #print(f"{axis_positions=}")
        c1 = np.repeat(axis_positions[:, np.newaxis], N_particles, axis=1)
        c2 = np.repeat(axis_positions[np.newaxis, :], N_particles, axis=0)
        diff_matrix[dim, :, :] = c1 - c2

    return diff_matrix


def zero_forces(particle_positions, particle_distances_arr):
    net_force = 0.0 * particle_positions
    return net_force


def periodic_bcs(positions, box_length):
    """Apply periodic boundry conditions by subtracting L in the  'violating' components

    Args:
        posititions (array): position array
        velocities (array): velocities array
    """
    positions[positions > box_length / 2] = (
        (positions[positions > box_length / 2] + box_length / 2) % box_length
    ) - box_length / 2
    positions[positions < -box_length / 2] = (
        (positions[positions < -box_length / 2] - box_length / 2) % box_length
    ) + box_length / 2
    return positions


def euler_position(x, v, h):
    "First order Euler approximation returns a position"
    return x + v * h


def euler_velocity(v, h, net_forces):
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


# def verlet_position(x, v, h, m, net_forces):
#     "Second order Verlet approximation returns a position"
#     return x + v * h + h * h * net_forces/ (2 * m)

# def verlet_velocity(x, v, h, m, net_forces):
#     "Second order Verlet approximation returns a velocity"
#     net_forces_forward = 
#     return v + h * h * 

class verlet():
    "Class implementing everything relating to the Verlet algorithm. Output is"
    "position and velocity at time t + h"
    def __init__(self, x, v, h, m, net_forces):
        self.x = x; self.v = v; self.h = h; self.m = m; self.net_forces = net_forces 

    def position(self):
        self.new_positions =  self.x + self.v * self.h + self.h*self.h * self.net_forces/ (2 * self.m) 
        return 0;
    
    def new_forces(self):
        self.position()
        particle_distances = sp.spatial.distance.cdist(self.new_positions, self.new_positions)
        self.net_new_forces = forces(self.new_positions, particle_distances)
        return 0;

    def velocity(self):
        self.new_forces()
        self.new_velocity =  self.v + (self.h*self.h / (2 * self.m)) * (self.net_new_forces + self.net_forces)
        return 0;

    def get_positions_velocity(self):
        """Returns the positions and velocity in one go. Output is a tuple with (positions, velocity)."""
        self.velocity()
        return self.new_positions, self.new_velocity
        
    def get_kinetic_energy(self):
        """Returns kinetic energy T = 1/2mv**2"""
        v_norm = np.linalg.norm(self.v, axis=1)
        self.kinetic_energy = 0.5 * self.m * v_norm**2
        #print(self.kinetic_energy)
        return self.kinetic_energy

    def get_potential_energy(self):
        """Returns potential energy (Lennard-Jones potential)"""
        particle_distances = sp.spatial.distance.cdist(self.x, self.x)
        potential_energy_square_array = lennard_jones_natural(particle_distances)
        potential_energy = potential_energy_square_array[~np.isnan(potential_energy_square_array)] #filter out distances to self
        #print(potential_energy)
        return np.sum(potential_energy, axis=0) #only one column is needed i think

def time_step(positions, velocities, h, L):

    # First look for smallest distances within each array to apply potential to then update whole diagram
    particle_distances = sp.spatial.distance.cdist(positions, positions)
    particle_forces = forces(positions, particle_distances)

    positions_new = euler_position(positions, velocities, h)
    velocities_new = euler_velocity(velocities, h, particle_forces)

    positions_new = periodic_bcs(positions_new, L) # apply bcs


    return positions_new, velocities_new 

def time_step_verlet(positions, velocities, h, L):
    """Same as above function except it used the Verlet algorithm"""

    particle_distances = sp.spatial.distance.cdist(positions, positions)
    particle_forces = forces(positions, particle_distances)

    verlet_onestep = verlet(positions, velocities, h, 1, particle_forces)
    kinetic_energy = verlet_onestep.get_kinetic_energy()
    potential_energy = verlet_onestep.get_potential_energy()
    positions_new, velocities_new = verlet_onestep.get_positions_velocity()

    positions_new = periodic_bcs(positions_new, L) # apply bcs

    return positions_new, velocities_new, kinetic_energy, potential_energy


def time_loop(initial_positions, initial_velocities, h, max_time, L):
    N_particles = np.shape(initial_positions)[0]
    N_timesteps = int(max_time / h)
    N_dimensions = np.shape(initial_positions)[1]

    particle_positions = initial_positions
    particle_velocities = initial_velocities
    results_positions = np.zeros((N_timesteps, N_particles, N_dimensions))
    results_velocities = np.zeros((N_timesteps, N_particles, N_dimensions))
    results_energies = np.zeros([N_timesteps, N_particles, 2]) #3rd dimension 1 for T 2 for V so that E = T + V is np.sum(..., axis=2)
    results_positions[0, :, :] = initial_positions
    results_velocities[0, :, :] = initial_velocities

    for i in range(0, N_timesteps):
        particle_positions, particle_velocities, results_energies[i,:, 0], results_energies[i,:, 1] = time_step_verlet(particle_positions, particle_velocities, h, L) 
        #print(results_energies[i, :, 0])
        results_positions[i,:,:] = particle_positions
        results_velocities[i,:,:] = particle_velocities


    return results_positions, results_velocities, results_energies



h=0.01
N = 10
dim = 2
max_time = 200
L = 20
v_max = 0.01


rng = np.random.default_rng(seed=test_seed)
x_0 = rng.uniform(low = -L/2, high = L/2, size = (N, dim))
v_0 = rng.uniform(low = -v_max, high = v_max, size = (N, dim))


#x_0 = np.array([[-0.9 * L, 0.90 * L], [0.3 * L, -0.10 * L]])
#v_0 = np.array([[0.0, -0.10], [-0.00, 0.10]])

#x_0 = np.array([[0.51 * L, 0.4 * L], [0.49 * L, 0.3 * L]])
#v_0 = np.array([[0.09, 0], [-0.09*3, 0]])
loop_results_x, loop_results_v, loop_results_e = time_loop(x_0, v_0, h, max_time, L)

n_particles = np.shape(x_0)[0]
# n-steps, n-particle, dimension
#plt.plot(loop_results_x[:,:,0], loop_results_x[:,:,1], marker='x')

# plt.xlim(0, 20)
# plt.ylim(0, 20)
# plt.# show()
# # plt.xlim(0, 20)
# # plt.ylim(0, 20)
# # plt.# show()
# plt.show()

# print(f'{np.shape(loop_results_x)}')
# print(loop_results_x)



""" 
hieronder de animatie meuk

"""



import matplotlib.animation as animation





def update(frame):
    # for each frame, update the data stored on each artist.
    multiplier = 1000
    frames_in_view = 500
    colors = ['red', 'blue', 'yellow', 'pink', 'green']
    number_of_colors = len(colors)

    for i in range(0,n_particles):
        bottom_line = ax.plot(loop_results_x[:,i,0], loop_results_x[:,i,1], marker='o', linestyle='none', c=colors[i%number_of_colors])[0]
        bottom_line.set_xdata(loop_results_x[:,i,0][multiplier*frame - frames_in_view:multiplier*frame])
        bottom_line.set_ydata(loop_results_x[:,i,1][multiplier*frame - frames_in_view:multiplier*frame])

    return (scat, bottom_line)



fig, ax = plt.subplots()

#ax.set_xlim([-L,L])
#ax.set_ylim([-L,L])

#scat = ax.scatter(loop_results_x[:,0,0], loop_results_x[:,0,1], s=0)
#plt.show()


# ax.legend()
# n_frames = np.shape(time_arr)[0]
# n_frames = 10
ani = animation.FuncAnimation(fig=fig, func=update,  interval=30, repeat=True)
plt.show()
