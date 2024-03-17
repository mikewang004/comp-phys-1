from main3 import * 


h = 0.1
N = 20
dim = 2
max_time = 200*2
L = 20
v_max = 1


# rng = np.random.default_rng(seed=test_seed)
# x_0 = rng.uniform(low = -L/2, high = L/2, size = (N, dim))
# v_0 = rng.uniform(low = -v_max, high = v_max, size = (N, dim))


# print(x_0)
# print(v_0)

# x_0 = np.array([[-0.9 * L, 0.90 * L], [0.3 * L, -0.10 * L]])
# v_0 = np.array([[0.0, -0.10], [-0.00, 0.10]])


x_0 = np.array(
    [
        [0.4 * L, 0.25*L], #x1, y1
        [0.7 * L, 0.15*L], #x2, y2
        [0.3 * L, 0.35 * L]
     ]
    )
x_0 = x_0 - L/2
v_0 = 1 + np.array([
    [0.19, 0.1],
    [-0.09, 0.1],
    [0.15, 0.1]
    ])

loop_results_x, loop_results_v, loop_results_E = time_loop(x_0, v_0, h, max_time, L)


#plt.scatter(loop_results_x[:, 0, 0], loop_results_x[:, 0, 1], marker = ".")
#plt.scatter(loop_results_x[:, 1, 0], loop_results_x[:, 1, 1], marker = ".")
#plt.scatter(loop_results_x[:, 2, 0], loop_results_x[:, 2, 1], marker = ".")
#plt.show()


#animate_results(loop_results_x[:, :2, 0], loop_results_x[:, :2, 1], frame_interval = 0.1)

