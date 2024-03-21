from main4 import *

# Testing file for energy plots
# Note for sim.results.energies
# 0th dim encodes timestep, 1st particle number, 2nd:
# 0 is kinetic energy, 1 is potential energy
# Recommended parameters as follows for resp. gas liquid solid:

#density = 0.3; temperature = 3.0;
density = 0.8; temperature = 1.0;
#density = 1.2; temperature = 0.5;
# Eq energy according to Maja should be about 80  
L = 20 #not relevant anymore
h = 0.01
max_time = 50
#density = 0.4; temperature = 1.424

x_0 = np.array(
    [[0.3 * L, 0.51 * L], 
     [0.7 * L, 0.49 * L], 
     [0.1 * L, 0.9 * L], 
     [0.4 * L, 0.1 * L]]
)
v_0 = 10 * np.array(
    [
        [0.09, -0.00],
        [-0.09, 0.00],
        [0.09, -0.00],
        [-0.09, 0.00],
    ]
)


def main():
    time_array = np.arange(0, max_time, int(max_time / h))
    sim = Simulation(
        L,
        h,
        max_time,
        animate=False,
        method="verlet",
        density=density,
        temperature=temperature,
    )
    # plt.plot(np.linalg.norm(sim.results.velocities[:, 0, :], axis =1), label = "velocity, norm")
    plt.plot(np.nansum(sim.results.energies[:, :, 0], axis=1), label="kinetic")
    plt.plot(np.nansum(sim.results.energies[:, :, 1], axis=1), label="potential")
    plt.plot(
        np.nansum(np.nansum(sim.results.energies[:, :, :], axis=1), axis=-1),
        label="total",
    )
    plt.legend()
    plt.title("total energy")
    plt.xlabel("time")
    plt.ylabel("energy")
    # plt.ylim(-10, 500)
    plt.show()


if __name__ == "__main__":
    main()
