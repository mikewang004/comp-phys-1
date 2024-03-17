# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants as spc
import matplotlib.animation as animation

# Define constants

epsilon = 1; sigma = 3.405 #Angstrom 
test_seed = 100

# Note nabla U = 4 * epsilon (( -12 * sigma**12 * r**(-13)) - 6*sigma**6 * r**(-7))


def lennard_jones_natural(dists_nat):
    "Returns Lennard-Jones potential as given. r_natural is the distance in rescaled (natural) units"
    return 4 * ((dists_nat) ** -12 - (dists_nat) ** -6)

def nabla_lennard_jones_natural(dists_nat):
    "Returns dau Lennard-Jones potential / dau r_natural as given. r_natural is the distance in rescaled (natural) units"
    return -4 * (12 * dists_nat ** -13.0 - 6 * dists_nat ** -7.0)

def forces2(particle_distances_arr, norm_diff_matrix, L, dimensions):
    """return net force array for all particles

    Args:
        particle_positions (arr): positions in N-d for the particles
        particle_distances_arr (arr): N x N array
    """
    # number of columns in the positions array corresponds to the dimension

    # calculate diff matrix across all dimensions

    # assuming particle_distances_arr has identical ordering but probably could be more general
    diagonals = np.eye(
        np.shape(particle_distances_arr)[0], np.shape(particle_distances_arr)[1], dtype=bool
    )
    forces_magnitudes = np.empty((np.shape(particle_distances_arr)))
    forces_magnitudes[~diagonals] = 4 * (
        -12 * particle_distances_arr[~diagonals] ** -13.0
        + 6 * particle_distances_arr[~diagonals] ** -7.0
    )

    net_force = np.zeros((N_particles, dimensions))
    repeated_force_magnitudes = np.repeat(
        forces_magnitudes[np.newaxis, :, :], dimensions, axis=0
    )
    # sum over other particles, '-1' for -nabla  V.
    net_force = -np.sum(repeated_force_magnitudes * norm_diff_matrix, axis=1).T

    return net_force


def forces(particle_positions, particle_distances_arr, epsilon = epsilon):
    """Rewrite of above function. Implementation of 12-6 potential""" 
    """U(r) = 4 (r**(-12) - r**(-6)) with r reduced distance"""
    #TODO fix for 3d
    particle_distances = sp.spatial.distance.cdist(particle_positions, particle_positions)
    # Calculate force corresponding to distance
    particle_pot_force = -1 * nabla_lennard_jones_natural(particle_distances)
    particle_pot_force_sum = np.nansum(particle_pot_force, axis = 1)
    print(particle_pot_force)
    test = np.repeat(particle_pot_force_sum[..., np.newaxis], 2, axis = -1)
    return test

def forces(particle_positions, particle_distances_arr):
    """Rewrite of above function. Implementation of 12-6 potential""" 
    """U(r) = 4 (r**(-12) - r**(-6)) with r reduced distance"""
    #First calculate distances per dimension and in total 
    # Note dimension encoded in last axis of array 
    dim = particle_positions.shape[-1]; size = particle_positions[0]
    particle_distances = sp.spatial.distance.cdist(particle_positions, particle_positions)
    particle_distances_dim = np.zeros([zero, size, size])
    # Calculate particle distances on vector-element-level
    for i in range (0, dim):
        particle_distances_dim[i] = particle_positions[:, i] - particle_positions.transpose[:, i]

    # Apply 12-6 potential 


def get_diff_and_dist(particle_positions, L, dimensions, N_particles):
    diff_matrix = calc_diff_matrix(particle_positions, dimensions, N_particles)

    # apply minimal image convention
    diff_matrix_min_image = (diff_matrix + L / 2) % L - L / 2
    particle_distances_arr = np.linalg.norm(diff_matrix_min_image, axis=0)

    return particle_distances_arr, diff_matrix_min_image 


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
    diagonals = np.eye(N_particles, N_particles, dtype=bool)
    # sum the squares of elements across all dimensions
    norm = np.linalg.norm(diff_matrix, axis=0)
    diff_matrix_inv_norm[~diagonals] = 1 / norm[~diagonals]

    # do the normalization
    diff_matrix[:, :, :] = diff_matrix[:, :, :] * (
        np.repeat(diff_matrix_inv_norm[np.newaxis, :, :], dimensions, axis=0)
    )
    return diff_matrix


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
        c1 = np.repeat(axis_positions[:, np.newaxis], N_particles, axis=1)
        c2 = np.repeat(axis_positions[np.newaxis, :], N_particles, axis=0)
        diff_matrix[dim, :, :] = c1 - c2

    return diff_matrix

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
    ) - box_length / 2
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
    return v + net_forces * h


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
        self.new_velocity =  self.v + (0.5/self.m) * self.h**2  * (self.net_new_forces + self.net_forces)
        return 0;

    def get_positions_velocity(self):
        """Returns the positions and velocity in one go. Output is a tuple with (positions, velocity)."""
        self.velocity()
        return self.new_positions, self.new_velocity
        
    def get_kinetic_energy(self):
        """Returns kinetic energy T = 1/2mv**2"""
        v_norm = np.linalg.norm(self.v, axis=1)
        self.kinetic_energy = 0.5 * self.m * v_norm**2
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
        results_positions[i,:,:] = particle_positions
        results_velocities[i,:,:] = particle_velocities

    return results_positions, results_velocities, results_energies


def animate_results(input_x, input_y, view_size=10, frame_interval=100, trailing_frames=1):
    fig, ax = plt.subplots()
    ax.set_xlim([-view_size, view_size])
    ax.set_ylim([-view_size, view_size])

    n_particles = np.shape(input_x)[1]
    n_frames = np.shape(input_x)[0] + 1
    lines = []
    plt.grid()
    # set up first frame for plotting and construct all lines
    for i in range(0, n_particles):
        frame = 0
        plotline = ax.plot(
            input_x[frame, i], input_y[frame, i], marker="o", linestyle="", markersize=2
        )
        lines.append(plotline[0])

    def update(frame):
        for i in range(0, len(lines)):
            line = lines[i]
            trailing_frame = max(0, frame - trailing_frames)
            line.set_xdata(
                input_x[trailing_frame:frame, i],
            )
            line.set_ydata(input_y[trailing_frame:frame, i])
        return lines

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=n_frames,
        interval=frame_interval,
        repeat=True,
        cache_frame_data=False,
    )

    plt.show()


def animate_quiver(
    positions_x,
    positions_y,
    vec_x,
    vec_y,
    view_size=10,
    frame_interval=10,
    arrow_scaling=1,
):
    # n_particles = np.shape(arrow_positions)[1]
    n_frames = np.shape(positions_x)[0]
    print(f"{n_frames=}")

    fig2, ax = plt.subplots(1, 1)
    quiver = ax.quiver(
        positions_x[0, :],  # 0 for first frame
        positions_y[0, :],
        vec_x[0, :],
        vec_y[0, :],
        pivot="tail",
        # color="r",
    )

    ax.set_xlim(-view_size, view_size)
    ax.set_ylim(-view_size, view_size)

    def update_quiver(n_frame, quiver, positions_x, positions_y, vec_x, vec_y):
        """updates the horizontal and vertical vector components by a
        fixed increment on each frame
        """
        print(f"{n_frame=}")
        U = arrow_scaling * vec_x[n_frame, :]
        V = arrow_scaling * vec_y[n_frame, :]

        positions = np.transpose(np.vstack((positions_x[n_frame, :], positions_y[n_frame, :])))
        print(f"{np.shape(positions)=}")
        quiver.set_offsets(positions)
        quiver.set_UVC(U, V)

        return (quiver,)

    anim = animation.FuncAnimation(
        fig2,
        update_quiver,
        fargs=(quiver, positions_x, positions_y, vec_x, vec_y),
        interval=frame_interval,
        blit=False,
        frames=n_frames,
    )
    plt.show()

    return

