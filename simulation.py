from main4 import *

# Following parameters generate a heart at about 150-200 time

L = 20
h = 0.02
max_time = 550

x_0 = np.array([[0.3 * L, 0.51 * L, 0.5 * L], [0.7 * L, 0.49* L, 0.1*L]])
v_0 = np.array(
    [
        [0.09, -0.00, 0.0],
        [-0.09, 0.00, 0.0],
    ]
)

def main():
    sim = simulation(L, h, max_time, x_0, v_0, True)

if __name__ == "__main__":
    main()
