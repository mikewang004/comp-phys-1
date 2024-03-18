from main4 import *

# Following parameters generate a heart at about 150-200 time

L = 10
h = 0.02
max_time = 550

x_0 = np.array([[-0.2 * L, 0.01 * L], [0.2 * L, -0.01 * L]])
v_0 = np.array(
    [
        [0.09, -0.00],
        [-0.09, 0.00],
    ]
)

def main():
    sim = simulation(L, h, max_time, x_0, v_0, False) #disable if working from simulation.py

if __name__ == "__main__":
    main()
