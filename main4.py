import numpy as np
from scipy import constants as spc
from visplot import *
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO figure out what to do about spc Boltzmann

m_ar = 6.6e-26
sigma = 3.405e-10
epsilon = 1.654e-21
# density = 0.3; temperature = 3.0;
# density = 0.8; temperature = 1.0;
# density = 1.2 ; temperature = 1


def lennard_jones_potential(r_nat):
    return 4 * (r_nat**-12 - r_nat**-6)


def get_e_target(n_particles, temperature):
    return (n_particles - 1) * (3 / 2) * temperature * spc.Boltzmann / epsilon


class Box:
    def __init__(
        self,
        box_length=1,
        particle_positions=None,
        particle_velocities=None,
        radial_distances = None,
        density=4,
        temperature=100,
    ):
        self.box_length = box_length
        self.density = density
        self.temperature = temperature
        self.positions = particle_positions
        self.radial_distances = radial_distances 
        self.velocities = particle_velocities
        self.positions_lookahead = None  # is this fair?
        if particle_positions is None:
            self.n_particles = None
            self.n_dimensions = None
        else:
            self.n_particles = np.shape(self.positions)[0]
            self.n_dimensions = np.shape(self.positions)[1]
        return

    def step_forward_euler(self, h):
        self.positions = self.positions + h * self.velocities
        self.velocities = self.velocities + h * self.get_forces()
        self.kinetic_energies = 0.5 * np.linalg.norm(self.velocities, axis=1) ** 2
        self.potential_energies = (
            self.get_potential_energies() * 0.5
        )  # Note imposed factor 0.5 for double counting
        return 0

    def step_forward_verlet(self, h):
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

    def apply_bcs(self):
        """Apply periodic boundry conditions by subtracting L in the 'violating' components
        Args:
            posititions (array): position array
            velocities (array): velocities array
        """
        self.positions[self.positions > self.box_length / 2] = (
            (self.positions[self.positions > self.box_length / 2] + self.box_length / 2)
            % self.box_length
        ) - self.box_length / 2
        self.positions[self.positions < -self.box_length / 2] = (
            (self.positions[self.positions < -self.box_length / 2] - self.box_length / 2)
            % self.box_length
        ) - self.box_length / 2
        return 0

    def get_forces(self):
        connecting_vectors = self.get_connecting_vectors()
        connecting_vectors = self.apply_min_im_convention(connecting_vectors)
        self.radial_distances = self.get_radial_distances(connecting_vectors)
        self.force_magnitudes = self.get_force_magnitudes(self.radial_distances)

        # normalize the vectors
        connecting_vectors = self.normalize_connecting_vectors(connecting_vectors)
        force_per_particle = connecting_vectors * (
            np.repeat(self.force_magnitudes[np.newaxis, :, :], self.n_dimensions, axis=0)
        )
        force_per_particle[np.isnan(force_per_particle)] = 0

        forces = np.sum(force_per_particle, axis=1)
        forces = np.transpose(forces)
        return forces

    def generate_velocities(self):
        # velocities = np.zeros((self.n_particles, self.n_dimensions))
        # velocities = np.exp(-1 * np.random.normal(size = (self.n_particles, self.n_dimensions)**2 / (2 * sp.constants.Boltzmann * self.temperature)))
        sigma = np.sqrt(2 * spc.Boltzmann * self.temperature / epsilon)
        self.velocities = np.random.normal(
            scale=sigma, size=(self.n_particles, self.n_dimensions)
        )

        return 0

    def rescale_velocities(self):
        # TODO does not work as expected
        e_target = get_e_target(self.n_particles, self.temperature)
        # Compare e_target to current kin en
        total_kin_en = np.nansum(self.kinetic_energies)
        print("e_target resp total kin en as follows")
        print(e_target)
        print(total_kin_en)
        if np.abs(total_kin_en - e_target) > 10 * np.std(e_target):
            labda = np.sqrt(e_target * 2 / total_kin_en)
            print(labda)
            self.velocities = labda * self.velocities
        return 0

    def generate_particle_positions(self):
        """Generates fcc-unit cells. Note code assumes three dimensions."""
        self.n_dimensions = 3
        self.n_particles = 108
        max_cube_counter = 3
        atoms_per_unit_cell = 4
        # cell_length = 4.0 / self.density ** (1 / 3)  # 4 particles in unit cell
        cell_length = self.box_length / max_cube_counter # 3 

        self.positions = np.zeros((self.n_particles, self.n_dimensions))
        # Generate one cell
        single_cell = -self.box_length /2 + np.array(
            [
                [0, 0, 0],
                [0.5 * cell_length, 0.5 * cell_length, 0],
                [0, 0.5 * cell_length, 0.5 * cell_length],
                [0.5 * cell_length, 0, 0.5 * cell_length],
            ]
        )
        # Generate full cube
        position_counter = 0
        a = 0
        for i in range(0, max_cube_counter):
            for j in range(0, max_cube_counter):
                for k in range(0, max_cube_counter):
                    self.positions[position_counter : position_counter + atoms_per_unit_cell , :] = (
                        single_cell
                        + np.array([k * cell_length, j * cell_length, i * cell_length])
                    )
                    position_counter = position_counter + atoms_per_unit_cell 
        return self.positions

    def get_pressure_avg_term(self):
        # First calculate average over all pairs
        # TODO verify if this works 
        avg_term = 0
        #for i in range(1, int(self.force_magnitudes.shape[0] / 2+1)):
        for i in range(1, self.radial_distances.shape[1]):
            avg_term = avg_term + np.trace(self.radial_distances, offset=i) * np.trace(
                self.force_magnitudes, offset=i
            )
        avg_term = 0.5 * avg_term
        return avg_term


class Results(object):
    def __init__(self):
        return


def get_x_component(object):
    return object[:, :, 0]


def get_y_component(object):
    return object[:, :, 1]

def get_z_component(object):
    return object[:, :, 2]

class Simulation:
    def __init__(self, system):
        self.system = system
        self.results = Results()
        self.stepping_function = self.system.step_forward_verlet
        #self.pressure_interval = 10 #Change this if needed 

    def run_simulation(self, h=0.1, max_time=1, method="verlet"):
        self.n_dimensions = self.system.n_dimensions
        self.n_particles = self.system.n_particles
        self.n_steps = int(max_time / h)

        self.results.positions = np.empty((self.n_steps, self.n_particles, self.n_dimensions))
        self.results.velocities = np.empty((self.n_steps, self.n_particles, self.n_dimensions))
        self.results.radial_distances= np.empty((self.n_steps, self.n_particles, self.n_particles))
        self.results.energies = np.empty(
            (self.n_steps, self.n_particles, 2)
        )  # 2nd axis: 0 for kin. energy 1 for pot. energy
        self.results.rad_dau_rad = np.empty((self.n_steps)) #for pressure sum-i sum-ij r_ij dau U(r_ij)/dau r
        self.results
        print(f'{np.shape(self.results.positions) =}')
        if method == "euler":
            self.stepping_function = self.system.step_forward_euler
        else:
            self.stepping_function = self.system.step_forward_verlet

        for i in tqdm(range(0, self.n_steps), desc="runnin"):
            self.stepping_function(h)
            self.system.apply_bcs()
            self.results.positions[i, :, :] = self.system.positions
            self.results.velocities[i, :, :] = self.system.velocities
            self.results.radial_distances[i, :, :] = self.system.radial_distances
            self.results.energies[i, :, 0] = self.system.kinetic_energies
            self.results.energies[i, :, 1] = self.system.potential_energies
            self.results.rad_dau_rad[i] = self.system.get_pressure_avg_term()
            #if i % 50 == 0:
            #    print(i)
            #     #TODO fix self system rescale velocities
            #    self.system.rescale_velocities()
        self.get_total_system_pressure(n_steps)

        # make a time array for easy plotting
        self.results.time = np.linspace(0, self.n_steps * h, num=self.n_steps)

    def get_total_system_kin_energy(self):
        return np.nansum(self.results.energies[:, :, 0], axis=1)

    def get_total_system_pot_energy(self):
        return np.nansum(self.results.energies[:, :, 1], axis=1)
    
    def get_total_system_pressure(self, n_steps):
        """Computes time-average pressure."""
        reshaped_pressure_avg_term = np.reshape(self.results.rad_dau_rad, (int(self.results.rad_dau_rad.shape[0]/ self.pressure_interval), self.pressure_interval))
        reshaped_pressure_avg_term = np.mean(reshaped_pressure_avg_term, axis = 1)
        self.results.pressure = self.system.temperature * self.system.density * (1 - 1/(3 * self.system.n_particles)) * reshaped_pressure_avg_term
        return 0;


    def animate_sim_results(self, frame_skip_multiplier=1):
        if self.n_dimensions == 2:
            animate_results(
                get_x_component(self.results.positions),
                get_y_component(self.results.positions),
                view_size=1.0 * self.system.box_length,
                frame_skip_multiplier=frame_skip_multiplier,
                trailing_frames=100000,
            )
        else:
            print("plotting projection in the xy plane")
            animate_results3d(
                get_x_component(self.results.positions),
                get_y_component(self.results.positions),
                get_z_component(self.results.positions),
                view_size=1.0 * self.system.box_length,
                frame_skip_multiplier=frame_skip_multiplier,
                trailing_frames=100000,
            )

    def plot_system_energy(self, which="all"):
        kin = self.get_total_system_kin_energy()
        pot = self.get_total_system_pot_energy()
        total = kin + pot
        plt.figure()
        if which == "all":
            make_xyplot(self.results.time, kin, plotname="kinetic energy")
            make_xyplot(self.results.time, pot, plotname="potential energy")
            make_xyplot(self.results.time, total, plotname="total")
        elif which == "total":
            make_xyplot(self.results.time, total, ylabel="total")
        else:
            pass
        plt.legend()

        plt.show()

        
    def generate_average_histogram(self, n_bins):
        final_box_length = self.system.box_length
        volume = final_box_length**3
        double_counting_factor = 1/2
        n_eval_frames= 100

        radius_bins = np.linspace(0, final_box_length, num=n_bins)
        bin_radial_width = radius_bins[1] - radius_bins[0]

        prefactor = double_counting_factor * 2 * volume / (4 * np.pi * self.system.n_particles * (self.system.n_particles - 1) )
        distance_norm = (radius_bins[0:-1] ** 2) * bin_radial_width

        # make a large flat array with all the radial distances 
        # from all steps and all particles are on one axis.
        selected_radial_distances = self.results.radial_distances[-1 -n_eval_frames:-1, :]
        flattened_radial_distances = selected_radial_distances.flatten()

        hist = np.histogram(flattened_radial_distances, bins=radius_bins)
        distances, frequencies = hist[1], hist[0]
        correlation_function = prefactor * frequencies/( distance_norm * n_eval_frames)
        return correlation_function, distances

    def plot_histogram(self, n_bins=900):
        # PAY ATTENTION. The edge alignment is only correct 
        # this way because generate_average_histogram only gives back bar boundries
        hist = self.generate_average_histogram(n_bins)
        plt.figure()
        plt.xlabel('radial distance $\\tilde{r}$')
        plt.ylabel('Correlation function g($\\tilde{r}$)')
        plt.stairs(hist[0], hist[1])
        plt.show()



L = 3.5
h = 0.0001
max_time = 500 * h
method = "verlet"
density = 10
temperature = 1000000

# x_0 = np.array(
#     [[0.3 * L, 0.51 * L], [0.7 * L, 0.49 * L], [0.1 * L, 0.9 * L], [0.4 * L, 0.1 * L]]
# )
# v_0 = np.array(
#     [
#         [0.09, -0.00],
#         [-0.09, 0.00],
#         [0.09, -0.00],
#         [-0.09, 0.00],
#     ]
# )
testbox1 = Box(
    box_length=L,
    density=density,
    temperature=temperature,
    particle_positions= x_0,
    particle_velocities= v_0
)
sim1 = Simulation(testbox1)
# np.savetxt("test.csv", sim1.results.energies[:, 0, :])


def main():
    #sim1.system.generate_particle_positions()
    #sim1.system.generate_velocities()
    print(f'{sim1.system.n_particles=}')
    print(f'{sim1.system.n_dimensions=}')
    # print(f'{np.shape(sim1.system.velocities)=}')
    print('vels')
    sim1.run_simulation(h=h, max_time=max_time, method=method)
    sim1.animate_sim_results(frame_skip_multiplier=1)
    sim1.plot_histogram(n_bins=90)


    # print(f'{np.shape(sim1.results.radial_distances) =}')
    # a = sim1.get_total_system_kin_energy()
    # print(f'{a=}')
    # sim1.plot_system_energy( which='all')
    print("Hello World!")


if __name__ == "__main__":
    main()
