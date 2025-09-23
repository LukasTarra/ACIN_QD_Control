"""import packages"""
import numpy as np
import matplotlib.pyplot as plt
from getparameters import get_parameters


# define basis vectors
N_STATES = 6 #number of states
if N_STATES != 6:
    raise ValueError("N_states must be 6 for a six-level system")
ground_state = np.zeros((N_STATES,1))
ground_state[0] = 1
exciton_x = np.zeros((N_STATES,1))
exciton_x[1] = 1
exciton_y = np.zeros((N_STATES,1))
exciton_y[2] = 1
dark_exciton_x = np.zeros((N_STATES,1))
dark_exciton_x[3] = 1
dark_exciton_y = np.zeros((N_STATES,1))
dark_exciton_y[4] = 1
biexciton = np.zeros((N_STATES,1))
biexciton[5] = 1
