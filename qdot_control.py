# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, vmap, random # , profiler
import jax.debug as jdebug
from jax.lax import scan, fori_loop
from functools import partial
from getparameters import ParametersQD, get_parameters_QD, get_default_pulse
import pdb
import warnings

from utilities import *
from qdot_simulation import *

# broader shell output before linebreaks for debugging 
jnp.set_printoptions(linewidth=300, edgeitems=10)

# """global variables"""
# NOMINAL_PARAMETERS = get_parameters_QD()
# HBAR = NOMINAL_PARAMETERS.hbar # Planck constant in weird units
# ZERO_TIME_FUNCTION = lambda t: 0.0*t
    

def main():
    # Load a timer
    timer = Timer()
    
    # Load QD parameters
    par_QD = get_parameters_QD()
    # Choose initial and target state (G, X_H, X_V, D_H, D_V, B)
    init_state = "G"
    target_state = "D_H"
    # no. of Trajectories for optimization & simulation
    number_trajectories_opt = 30
    number_trajectories_sim = 30
    # Polarization overlaps (e_H * e_L, e_V * e_L)
    polarization_overlaps = jnp.array([1,0])
    omega_X = par_QD.hbar_omega_X/par_QD.hbar

    # obtain default pulses and their pulsewidths
    time_until_init_pulse = 30 #ps
    init_pulse_default = get_default_pulse("initialization", time_until_init_pulse + 0, omega_X)
    init_pulse_function = init_pulse_default.get_chirped_pulse_function()
    init_pulse_parameters = init_pulse_default.get_pulse_parameters_dict()
    storage_pulse_default = get_default_pulse("storage", time_until_init_pulse + 70, omega_X)
    storage_pulse_function = storage_pulse_default.get_chirped_pulse_function()
    storage_pulse_parameters = storage_pulse_default.get_pulse_parameters_dict()
    retrieval_pulse_default = get_default_pulse("retrieval", time_until_init_pulse + 1420, omega_X)
    retrieval_pulse_function = retrieval_pulse_default.get_chirped_pulse_function()
    retrieval_pulse_parameters = retrieval_pulse_default.get_pulse_parameters_dict()
    
    # set up the simulation-relevant JAX matrices etc.
    dt_sim = 0.2e-2* par_QD.hbar/par_QD.E_B* 2*jnp.pi
    N_Hamiltonian_steps = 2
    time_array = jnp.arange(0, 1520, dt_sim)    
    H_0_tilde_eff_j, L_operators_j, Ldag_operators_j, sum_LdagL_operators_j, psi_0_j, psi_T_j, H_c_tilde_real_j, H_c_tilde_imag_j, I_imag_j, I_H_0_tilde_eff_j, I_H_c_tilde_real_j, I_H_c_tilde_imag_j, U_rotating_frame_j = jax_sim_setup(init_state, target_state, polarization_overlaps, time_array, par_QD)

    # define the baseline input array
    # hard factor in front of input
    input_factor = 1
    input_array = input_factor * jnp.array( init_pulse_function(time_array) + storage_pulse_function(time_array) + retrieval_pulse_function(time_array) )
    input_array_real = jnp.real(input_array)
    input_array_imag = jnp.imag(input_array)

    # simulate the system
    number_collapse = L_operators_j.shape[0]
    number_steps = len(time_array)
    dW_real_j, dW_imag_j = create_jax_noise_traj_arrays(20, number_collapse, number_trajectories_sim, number_steps, dt_sim)
    timer.start()
    traj = np.array( sim_forward_vmap(input_array_real, input_array_imag, dW_real_j, dW_imag_j, psi_0_j, I_H_0_tilde_eff_j, L_operators_j, Ldag_operators_j, sum_LdagL_operators_j, I_H_c_tilde_real_j, I_H_c_tilde_imag_j, dt_sim, I_imag_j, N_Hamiltonian_steps) )
    _ = timer.stop()

    # Create QDSimResults object and plot results
    sim_results = QDSimResults(traj, time_array, par_QD, input_array_real)
    sim_results.plot_control_field("Control Field (Real Part)")
    sim_results.plot_mean_population_trajectories()
    sim_results.plot_mean_dressed_population_trajectories()


if __name__ == "__main__":

    main()

