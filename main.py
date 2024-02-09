import numpy as np
import matplotlib.pyplot as plt

#Define constants

e_kb = 119.8 #K
sigma = 3.405 #A


def lennard_jones(epsilon, sigma, r):
    "Returns Lennard-Jones potential as given"
    return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

