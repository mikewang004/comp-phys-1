import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.constants as spc
#Define constants

e_kb = 119.8 #K
#epsilon = e_kb * sp.constants.Boltzmann
#sigma = 3.405 #A
epsilon = 1; sigma = 1;
h = 0.01 #timestep 
N = 6 #number of particles
dim = 3 #number of spacelike dimensions
box_length = 20 #m
m = 6.6 * 10**-26
natural_constant =  (m *sigma**2) / epsilon
v_start = sigma / np.sqrt(natural_constant)
max_time = 100


test_seed = 100

#Note nabla U = 4 * epsilon (( -12 * sigma**12 * r**(-13)) - 6*sigma**6 * r**(-7))


def lennard_jones_natural(dists_nat):
    "Returns Lennard-Jones potential as given. r_natural is the distance in rescaled (natural) units"
    return 4 * epsilon * ((dists_nat)**-12 - (dists_nat)**-6)

def nabla_lennard_jones_natural(r_natural):
    return 4*(-12 * r_natural**-12.0 + 6 * r_natural**-7.0)

def potential(x, dists_nat, potential_function, ):
    """Assuming distance-only potential given by dU/dr * x_vector / r. Note potential here assumed to be 
    dU/dr i.e. nabla is assumed to be already applied to the potential."""
    return potential_function(dists_nat, epsilon, sigma) * (x / dists_nat)

def force_natural(x, dists_nat, potential_function = nabla_lennard_jones_natural):
    r_repeat = np.transpose(np.tile(dists_nat, (dim, 1)))
    return potential_function(r_repeat) 

def periodic_bcs(positions, velocities, box_length):
    """Apply periodic boundry conditions by subtracting L in the  'violating' components

    Args:
        posititions (array): position array
        velocities (array): velocities array
    """
    
    positions[positions > box_length] = positions[positions > box_length] - box_length
    positions[positions < 0] = positions[positions < 0] + box_length

    return positions

def euler_position(x, v, h):
    "First order Euler approximation returns a position"
    return x + v * h
    
def verlet_position(x, v, m, force_natural, h):
    #TODO nog potentieel verwerken 
    return x + h * v + h*h/(2*m) * force_natural

def verlet_velocity(x_next, v, m, force_natural_new, force_natural, h):
    return v + h/(2*m) * (force_natural_new + force_natural)

def verlet_algorithm(x, v, m, force_natural = force_natural, h= h):
    """Runs one timestep of the Verlet algorithm."""
    x_new = verlet_position(x, v, m, force_natural, h)
    force_new = force_natural
    
def energy(dists_nat,v_natural, potential=potential,):
    print('dists', np.shape(dists_nat), np.shape(v_natural))
    return np.sum(1/2 * epsilon * v_natural**2, axis=0) + np.sum(lennard_jones_natural(dists_nat), axis=0)

def euler_velocity(v, potential, h):
    "First order Euler approximation note potential requires a function"
    #TODO double check if velocity truely unitless 
    return v + potential * h


def time_step(x, v, h, potential=force_natural, L = box_length):
    #First look for smallest distances within each array to apply potential to then update whole diagram
    r_distances = sp.spatial.distance.cdist(x_0, x_0)
    r_distances[r_distances == 0] = np.nan
    r_min = (np.nanmin(r_distances, axis=0, keepdims=False)) #transpose possibly unnecessary 
    #Apply potential 
    pot_x = potential(x, r_min)
    #pot_x = x
    v = euler_velocity(v, pot_x, h)
    x = euler_position(x, v, h)
    x = periodic_bcs(x, v, L)
    system_energy = energy(r_distances,v)
    return x, v, system_energy 

def test():
    pass

def time_loop(x_0, v_0, h, max_time, potential = force_natural):
    x = x_0; v = v_0
    #Initialise positions-of-all-time array 
    x_all = np.zeros([x_0.shape[0], x_0.shape[1], int(max_time/h)])
    x_all[:, :, 0] = x_0
    for i in range(0, max_time):
        x, v, energy = time_step(x, v, h, potential)
        x_all[:, :, i] = x
    return x, v, x_all




rng = np.random.default_rng()
x_0 = rng.uniform(low = -10, high = 10, size = (N, dim))
v_0 = rng.uniform(low = -3, high = 3, size = (N, dim))

#time_step(x_0, v_0, h)
x, v, x_all = time_loop(x_0, v_0, h, max_time)
print(x_0)
energy_over_time = x_all[:,-1]
plt.figure()
plt.plot(energy_over_time)
plt.show()
print('energy', x_all[:,-1])



#Test plotting one particle 

print(x - x_0)

for j in range(0, N):
    plt.scatter(x_all[j, 0, :], x_all[j, 1, :], marker=".")


plt.show()