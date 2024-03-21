from main4 import *


def main():

# Following parameters generate a heart at about 150-200 time
    box_length = 20
    L = box_length
    h = 0.01
    max_time = 50

    #x_0 = np.array([[0.3 * L, 0.51 * L, 0.5 * L], [0.7 * L, 0.49* L, 0.1*L]]) #<--- those are the heart parameters
    x_0 = -L/2 + np.array([[0.3 * L, 0.51 * L], [0.7 * L, 0.49 * L]])
    v_0 = np.array(
    [
        [0.09, -0.00],
        [-0.09, 0.00],
    ]
    )

    testbox1 = Box(
    density=density,
    temperature=temperature,
    particle_positions =  x_0, 
    particle_velocities = v_0,
    box_length = box_length
    )
    sim1 = Simulation(testbox1)
    sim1.run_simulation(h=h, max_time=max_time, method=method)

    # plt.scatter(sim1.results.positions[:, 0, 0], sim1.results.positions[:, 0, 1], c = 'blue', marker = ".")
    # plt.scatter(sim1.results.positions[0, 0, 0], sim1.results.positions[0, 0, 1], c = 'black', marker ="x")
    # plt.scatter(sim1.results.positions[:, 1, 0], sim1.results.positions[:, 1, 1], c = 'red', marker = ".")
    # plt.scatter(sim1.results.positions[0, 1, 0], sim1.results.positions[0, 1, 1], c = 'black', marker="x")
    # plt.xlabel("position")
    # plt.ylabel("position")
    # plt.title("Trajection path of two particles, t = 50 [unitless], h = 0.01")
    # #plt.savefig("traject-2d.pdf")
    # plt.show()

    plt.plot(np.linalg.norm(sim1.results.velocities[:, 0, :], axis = 1))
    plt.plot(1 + -1*np.linalg.norm(sim1.results.velocities[:, 1, :], axis = 1))
    plt.plot((1 + -1*np.linalg.norm(sim1.results.velocities[:, 1, :], axis = 1) + np.linalg.norm(sim1.results.velocities[:, 0, :], axis = 1)))
    plt.xlabel("time")
    plt.ylabel("momentum")
    plt.title("Momentum evaluation of two particles, t = 50 [unitless], h = 0.01")
    plt.savefig("momentum-2d.pdf")
    plt.show()

    

if __name__ == "__main__":
    main()
