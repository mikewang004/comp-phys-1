import numpy as np
from scipy import constants as spc
from visplot import *

# TODO figure out what to do about spc Boltzmann

m_ar = 6.6e-26
sigma = 3.405e-10
epsilon = 1.654e-21

def lennard_jones_potential(r_nat):
    return 4 * (r_nat**-12 - r_nat**-6)


def get_e_target(n_particles, temperature):
    return (n_particles - 1) * (3 / 2) * temperature * spc.Boltzmann/epsilon




class Box:
    def __init__(
        self,
        particle_positions=None,
        particle_velocities=None,
        density=4,
        temperature=100,
    ):
        self.density = density
        self.temperature = temperature
        self.n_dimensions = 3
        if particle_positions is None:
            self.positions = self.generate_particle_positions()
        else:
            self.positions = particle_positions
        self.positions_lookahead = None  # is this fair?
        self.n_particles = np.shape(self.positions)[0]
        self.n_dimensions = np.shape(self.positions)[1]
        if particle_velocities is None:
            self.velocities = self.generate_velocities()
        else:
            self.velocities = particle_velocities

    def step_forward_euler(self, h):
        self.positions = self.positions + h * self.velocities
        self.velocities = self.velocities + h * self.get_forces()
        self.kinetic_energies = 0.5 * np.linalg.norm(self.velocities, axis=1) ** 2
        self.potential_energies = (
            self.get_potential_energies() * 0.5
        )  # Note imposed factor 0.5 for double counting
        return 0

    def step_forward_verlet(self, h, first_step=False):
        forces_current = np.copy(self.get_forces())
        self.kinetic_energies = 0.5 * np.linalg.norm(self.velocities, axis=1) ** 2
        self.potential_energies = (
            self.get_potential_energies() * 0.5
        )  # Note imposed factor 0.5 for double counting
        # do a lookahead positions
        self.positions = (
            self.positions + h * self.velocities + h**2 / 2 * self.get_forces()
        )
        # calculate forces at new positions
        forces_lookahead = np.copy(self.get_forces())
        self.velocities = self.velocities + h / 2 * (forces_lookahead + forces_current)
        return 0

    def get_potential_energies(self):
        # print(lennard_jones_potential(self.radial_distances))
        return np.nansum((lennard_jones_potential(self.radial_distances)), axis=0)

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
        dim_tiled_norm = np.repeat(
            diff_matrix_norm[np.newaxis, :, :], self.n_dimensions, axis=0
        )
        diff_matrix = diff_matrix / dim_tiled_norm

        return diff_matrix

    def get_radial_distances(self, diff_matrix):
        radial_distances = np.linalg.norm(diff_matrix, axis=0)  # sum over dimensions
        return radial_distances

    def get_force_magnitudes(self, radial_distances):
        forces_magnitudes = 4 * (-12 * radial_distances**-13 + 6 * radial_distances**-7)
        return forces_magnitudes

    def apply_min_im_convention(self, diff_matrix):
        return (diff_matrix + L / 2) % L - L / 2

    def get_forces(self):
        connecting_vectors = self.get_connecting_vectors()
        connecting_vectors = self.apply_min_im_convention(connecting_vectors)
        radial_distances = self.get_radial_distances(connecting_vectors)
        self.radial_distances = radial_distances
        force_magnitudes = self.get_force_magnitudes(radial_distances)

        # normalize the vectors
        connecting_vectors = self.normalize_connecting_vectors(connecting_vectors)
        force_per_particle = connecting_vectors * (
            np.repeat(force_magnitudes[np.newaxis, :, :], self.n_dimensions, axis=0)
        )
        force_per_particle[np.isnan(force_per_particle)] = 0

        forces = np.sum(force_per_particle, axis=1)
        forces = np.transpose(forces)
        return forces

    def generate_velocities(self):
        # velocities = np.zeros((self.n_particles, self.n_dimensions))
        # velocities = np.exp(-1 * np.random.normal(size = (self.n_particles, self.n_dimensions)**2 / (2 * sp.constants.Boltzmann * self.temperature)))
        sigma = np.sqrt(2 * spc.Boltzmann * self.temperature)
        velocities = np.random.normal(
            scale=sigma, size=(self.n_particles, self.n_dimensions)
        )
        return velocities

    def rescale_velocities(self):
        # TODO does not work as expected
        e_target = get_e_target(self.n_particles, self.temperature)
        # Compare e_target to current kin en
        total_kin_en = np.nansum(self.kinetic_energies)
        print("e_target resp total kin en as follows")
        print(e_target)
        print(total_kin_en)
        if np.abs(total_kin_en - e_target) > 10 * np.std(e_target):
            labda = np.sqrt(e_target*2 / total_kin_en)
            print(labda)
            self.velocities = labda * self.velocities
        return 0

    def generate_particle_positions(self):
        """Generates fcc-unit cells. Note code assumes three dimensions."""
        n_particles = 108
        cell_length = 4.0 / self.density ** (1 / 3)  # 4 particles in unit cell
        positions = np.zeros((n_particles, self.n_dimensions))
        # Generate one cell
        single_cell = np.array(
            [
                [0, 0, 0],
                [0.5 * cell_length, 0.5 * cell_length, 0],
                [0, 0.5 * cell_length, 0.5 * cell_length],
                [0.5 * cell_length, 0, 0.5 * cell_length],
            ]
        )
        # Generate full cube
        max_cube_counter = 3
        position_counter = 0
        a = 0
        for i in range(0, max_cube_counter):
            for j in range(0, max_cube_counter):
                for k in range(0, max_cube_counter):
                    positions[position_counter : position_counter + 4, :] = (
                        single_cell
                        + np.array([k * cell_length, j * cell_length, i * cell_length])
                    )
                    position_counter = position_counter + 4
        return positions



class Results(object):
    def __init__(self):
        return


def get_x_component(object):
    return object[:, :, 0]


def get_y_component(object):
    return object[:, :, 1]


class Simulation:
    def __init__(self, system):
        self.system = system
        self.n_dimensions = system.n_dimensions
        self.n_particles = system.n_particles
        self.results = Results()

    def run_simulation_euler(self, h=0.1, max_time=1):
        n_steps = int(max_time / h)
        self.results.positions = np.empty((n_steps, self.n_particles, self.n_dimensions))
        self.results.velocities = np.empty((n_steps, self.n_particles, self.n_dimensions))

        for i in range(0, n_steps):
            # self.system.step_forward_euler(h)
            self.system.step_forward_euler(h)
            self.results.positions[i, :, :] = self.system.positions
            self.results.velocities[i, :, :] = self.system.velocities
        return 0

    def run_simulation_verlet(self, h=0.1, max_time=1, method="verlet"):
        n_steps = int(max_time / h)
        self.results.positions = np.empty((n_steps, self.n_particles, self.n_dimensions))
        self.results.velocities = np.empty((n_steps, self.n_particles, self.n_dimensions))
        self.results.energies = np.empty(
            (n_steps, self.n_particles, 2)
        )  # 2nd axis: 0 for kin. energy 1 for pot. energy
        if method == "euler":
            stepping_function = self.system.step_forward_euler
            print("we doin euler")
        else:
            stepping_function = self.system.step_forward_verlet
            print("we doin verlet")

        for i in range(0, n_steps):
            # self.system.step_forward_euler(h)
            stepping_function(h)
            self.results.positions[i, :, :] = self.system.positions
            self.results.velocities[i, :, :] = self.system.velocities
            self.results.energies[i, :, 0] = self.system.kinetic_energies
            self.results.energies[i, :, 1] = self.system.potential_energies
            if i % 500 == 0:
                print(i)
                #TODO fix self system rescale velocities
                #self.system.rescale_velocities()
        return 0


def simulation(
    L,
    h,
    max_time,
    x_0=None,
    v_0=None,
    animate=False,
    method="verlet",
    density=10,
    temperature=100,
):
    testbox1 = Box(x_0, v_0, density=density, temperature=temperature)
    sim1 = Simulation(testbox1)
    sim1.run_simulation_verlet(h=h, max_time=max_time, method=method)
    # np.savetxt("test.csv", sim1.results.energies[:, 0, :])

    if animate == True:
        animate_results(
            get_x_component(sim1.results.positions),
            get_y_component(sim1.results.positions),
            view_size=0.6 * L,
            frame_skip_multiplier=10,
            trailing_frames=100000,
        )
        plt.show()
    return sim1


L = 20
h = 0.02
max_time = 50

x_0 = np.array(
    [[0.3 * L, 0.51 * L], 
     [0.7 * L, 0.49 * L], 
     [0.1 * L, 0.9 * L], 
     [0.4 * L, 0.1 * L]]
)
v_0 = np.array(
    [
        [0.09, -0.00],
        [-0.09, 0.00],
        [0.09, -0.00],
        [-0.09, 0.00],
    ]
)


def main():
    #simulation(L, h, max_time, x_0, v_0)  # disable if working from simulation.py
    print("Hello World!")


if __name__ == "__main__":
    main()
