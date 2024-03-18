from main4 import *

# Testing file for energy plots 
# Note for sim.results.energies 
# 0th dim encodes timestep, 1st particle number, 2nd: 
# 0 is kinetic energy, 1 is potential energy
L = 10
h = 0.01
max_time = 150

x_0 = np.array([[0.3 * L, 0.51 * L], [0.7 * L, 0.49* L]])
v_0 = np.array(
    [
        [0.09, -0.00],
        [-0.09, 0.00],
    ]
)

def main():
    time_array = np.arange(0, max_time, int(max_time/h))
    sim = simulation(L, h, max_time, x_0, v_0, False)
    #plt.plot(np.linalg.norm(sim.results.velocities[:, 0, :], axis =1), label = "velocity, norm")
    plt.plot(sim.results.energies[:, 0, 0], label ="kinetic")
    plt.plot(sim.results.energies[:, 0, 1], label = "potential")
    #plt.plot(np.sum(sim.results.energies[:, 0, :], axis = 1), label = "total")
    plt.legend()
    plt.title("energy")
    plt.xlabel("time")
    plt.ylabel("energy")
    plt.show()

if __name__ == "__main__":
    main()






