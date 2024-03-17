import numpy as np
from visplot import *

class Box:
    def __init__(self, particle_positions, particle_velocities, box_length):
        self.positions = particle_positions
        self.previous_positions = particle_positions # is this fair?
        self.velocities = particle_velocities
        self.box_length = box_length
        self.n_particles = np.shape(particle_positions)[0]
        self.n_dimensions= np.shape(particle_positions)[1]
        
    def step_forward_euler(self, h):
        self.positions = self.positions + h*self.velocities
        self.velocities = self.velocities + h*self.get_forces()
        return

    def step_forward_verlet(self, h, first_step=False):
        if first_step:
            self.previous_positions = self.positions - h * self.velocities
        else:
            current_postisions = np.copy(self.positions)

            self.positions  =  2 * self.positions - self.previous_positions + h**2 * self.get_forces()
            self.previous_positions =  current_postisions
        return

        
    def get_connecting_vectors(self):
        diff_matrix = np.empty((self.n_dimensions, self.n_particles, self.n_particles))
        for dim in range(0, self.n_dimensions):
            axis_positions = self.positions[:, dim]
            c1 = np.repeat(axis_positions[:, np.newaxis], self.n_particles, axis=1)
            c2 = np.repeat(axis_positions[np.newaxis, :], self.n_particles, axis=0)
            diff_matrix[dim, :, :] = c1 - c2

        return diff_matrix

    def normalize_connecting_vectors(self, diff_matrix):

        # sum the squares of elements across all dimensions
        diff_matrix_norm = np.empty((self.n_particles, self.n_particles))
        norm = np.linalg.norm(diff_matrix, axis=0)
        diff_matrix_norm = norm

        # do the normalization
        dim_tiled_norm = np.repeat(diff_matrix_norm[np.newaxis, :, :], self.n_dimensions, axis=0)
        diff_matrix = diff_matrix / dim_tiled_norm

        return diff_matrix

    def get_radial_distances(self, diff_matrix):
        radial_distances = np.linalg.norm(diff_matrix, axis=0) # sum over dimensions
        return radial_distances

    def get_force_magnitudes(self, radial_distances):
        forces_magnitudes = 4 * (-12 * radial_distances**-13 + 6 * radial_distances**-7)
        return forces_magnitudes


    def get_forces(self):
        connecting_vectors = self.get_connecting_vectors() 
        radial_distances = self.get_radial_distances(connecting_vectors)
        force_magnitudes = self.get_force_magnitudes(radial_distances) 
        # normalize the vectors
        connecting_vectors = self.normalize_connecting_vectors(connecting_vectors)

        force_per_particle =  connecting_vectors * (np.repeat(force_magnitudes[np.newaxis, :, :], self.n_dimensions, axis=0))
        force_per_particle[np.isnan(force_per_particle)] = 0

        forces = np.sum(force_per_particle, axis=1)
        forces = np.transpose(forces)
        return forces

        
class Results(object):    
    def __init__(self) :
        pass

def get_x_component(object):
    return object[:,:,0]

def get_y_component(object):
    return object[:,:,1]

class Simulation:
    def __init__(self, system):
        self.system = system
        self.n_dimensions = system.n_dimensions
        self.n_particles = system.n_particles 
        self.results = Results()
        return

    def run_simulation_euler(self, h=0.1, max_time=1):
        n_steps = int(max_time/h)
        self.results.positions = np.empty((n_steps, self.n_particles, self.n_dimensions))
        self.results.velocities= np.empty((n_steps, self.n_particles, self.n_dimensions))

        for i in range(0, n_steps):
            # self.system.step_forward_euler(h)
            self.system.step_forward_euler(h)
            self.results.positions[i,:,:] = self.system.positions
            self.results.velocities[i,:,:] = self.system.velocities
        return

    def run_simulation_verlet(self, h=0.1, max_time=1):
        n_steps = int(max_time/h)
        self.results.positions = np.empty((n_steps, self.n_particles, self.n_dimensions))
        self.results.velocities= np.empty((n_steps, self.n_particles, self.n_dimensions))


        self.system.step_forward_verlet(h, first_step = True)
        self.results.positions[1,:,:] = self.system.positions
        self.results.velocities[1,:,:] = self.system.velocities

        for i in range(1, n_steps):
            # self.system.step_forward_euler(h)
            self.system.step_forward_verlet(h)
            self.results.positions[i,:,:] = self.system.positions
            self.results.velocities[i,:,:] = self.system.velocities
        
    


L = 20

x_0 = np.array([[-0.2 * L, 0.01 * L], 
                [0.2 * L, -0.01 * L]])
v_0 = np.array([[0.09, -0.00], 
                [-0.09, 0.00],
                ])

testbox = Box(x_0, v_0, L )
sim1 = Simulation(testbox)
sim1.run_simulation_verlet(h=0.01, max_time=100)
print((sim1.results.positions))
print(get_x_component(sim1.results.positions))

animate_results(get_x_component(sim1.results.positions), get_y_component(sim1.results.positions), view_size=0.6*L, frame_skip_multiplier=10,     trailing_frames = 100000)

# print(f'{testbox.particle_positions = }')
# print(testbox.step_forward_euler(0.1))
# print(f'{testbox.particle_positions = }')




