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

    

def forces(particle_positions, particle_distances_arr):
    """return net force array for all particles

    Args:
        particle_positions (arr): positions in N-d for the particles
        particle_distances_arr (arr): N x N array
    """
    dimensions = np.shape(particle_positions)[1] # number of columns in the positions array corresponds to the dimension
    N_particles = np.shape(particle_positions)[0]
    # the diff_matrix contains all possible combinations of vectors between a set of two particles
    diff_matrix = np.zeros((dimensions, N_particles,N_particles)) 

    # calculate diff matrix across all dimensions
    for dim in range(0, dimensions):
        axis_positions = particle_positions[:,dim]
        print(f'{axis_positions=}')
        c1 = np.repeat(axis_positions [:,np.newaxis], N_particles, axis=1)
        c2 = np.repeat(axis_positions [np.newaxis, :], N_particles, axis=0)
        diff_matrix[dim, :,:] = c1 -c2       

    # make a matrix to store the inverted norm of all diff vectors so we can normalize them
    diff_matrix_inv_norm  = np.zeros((N_particles, N_particles))
    diff_matrix_inv_norm[:,:] = 1/np.sqrt((np.sum(diff_matrix[:,:]**2, axis=0))) # sum the squares of elements across all dimensions

    # do the normalization    
    norm_diff_matrix = diff_matrix.copy()
    norm_diff_matrix[:,:,:] = norm_diff_matrix[:,:,:] * (np.repeat(diff_matrix_inv_norm[np.newaxis, :, : ], dimensions, axis=0))
    norm_diff_matrix[np.isnan(norm_diff_matrix)] = 0

    # assuming particle_distances_arr has identical ordering but probably could be more general
    forces_magnitudes = np.zeros((np.shape(particle_distances_arr)))
    forces_magnitudes[:,:] = 4*(-12 * particle_distances_arr[:,:]**-12.0 + 6 * particle_distances_arr[:,:]**-7.0) 
    forces_magnitudes[np.isnan(forces_magnitudes)] = 0


    net_force = np.zeros((N_particles, dimensions))
    repeated_force_magnitudes = np.repeat(forces_magnitudes[np.newaxis, :,: ], dimensions, axis=0)

    #sum over other particles
    net_force = np.sum(repeated_force_magnitudes * norm_diff_matrix, axis = 1).T
    
    return net_force

def zero_forces(particle_positions, particle_distances_arr):
    net_force = 0.0 * particle_positions
    return net_force
    

def periodic_bcs(positions, box_length):
    """Apply periodic boundry conditions by subtracting L in the  'violating' components

    Args:
        posititions (array): position array
        velocities (array): velocities array
    """
    positions[positions > box_length/2] = ((positions[positions > box_length/2] + box_length/2)% box_length)  - box_length/2
    positions[positions < -box_length/2] = ((positions[positions < -box_length/2] - box_length/2) % box_length)  + box_length/2
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
    particle_forces = 0.5* forces(positions, particle_distances)

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



h=2
N = 2
dim=2
max_time = 200
L = 20
v_max= 0.1


rng = np.random.default_rng()
# x_0 = rng.uniform(low = -L/2, high = L/2, size = (N, dim))
# v_0 = rng.uniform(low = -v_max, high = v_max, size = (N, dim))


x_0 = np.array([[-0.9 * L, 0.90 * L], [0.3 * L, -0.10 * L]])
v_0 = np.array([[0.0, -0.10], [-0.00, 0.10]])
loop_results_x, loop_results_v = time_loop(x_0, v_0, h, max_time, L)

n_particles= np.shape(x_0)[0]
# # n-steps, n-particle, dimension
# for i in range(0,n_particles):
#     plt.plot(loop_results_x[:,i,0], loop_results_x[:,i,1], marker='x')

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


fig, ax = plt.subplots()

ax.set_xlim([-L,L])
ax.set_ylim([-L,L])

scat = ax.scatter(loop_results_x[:,0,0], loop_results_x[:,0,1], s=0)


# ax.legend()


def update(frame):
    # for each frame, update the data stored on each artist.
    multiplier = 1 
    frames_in_view = 500
    colors = ['red', 'blue', 'yellow', 'pink', 'green']
    number_of_colors = len(colors)

    for i in range(0,n_particles):
        bottom_line = ax.plot(loop_results_x[:,i,0], loop_results_x[:,i,1], marker='o', linestyle='none', c=colors[i%number_of_colors])[0]
        bottom_line.set_xdata(loop_results_x[:,i,0][multiplier*frame - frames_in_view:multiplier*frame])
        bottom_line.set_ydata(loop_results_x[:,i,1][multiplier*frame - frames_in_view:multiplier*frame])

    return (scat, bottom_line)


# n_frames = np.shape(time_arr)[0]
# n_frames = 10
ani = animation.FuncAnimation(fig=fig, func=update,  interval=30, repeat=True)
plt.show()
ax.legend()



