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
from jaxopt import ScipyBoundedMinimize

from utilities import *
from qdot_simulation import *

# broader shell output before linebreaks for debugging 
jnp.set_printoptions(linewidth=300, edgeitems=10)

# """global variables"""
# NOMINAL_PARAMETERS = get_parameters_QD()
# HBAR = NOMINAL_PARAMETERS.hbar # Planck constant in weird units
# ZERO_TIME_FUNCTION = lambda t: 0.0*t

# @jit
def cost_function_target_state(traj: jnp.ndarray, psi_target: jnp.ndarray, t_array):
    """computes the trajectories deviation from the target state.
    inputs:
    - traj (jax.numpy array): resulting trajectories from simulation.
    - psi_target (jax.numpy array): target state (in real representation).
    - t_array (jax.numpy array): array of time values.
    returns:
    - cost value (scalar, integrated deviation from target state).
    """
    T = t_array[-1]
    projection = jnp.mean(jnp.sum(traj * traj - (traj * psi_target) * (traj * psi_target), axis=2), axis=0)
    cost = 1 / T * jnp.trapezoid(projection.squeeze(), x=t_array)

    return cost


def optimize_delta_p(
    pulse_id, 
    t_pulse_center, 
    omega_RF, 
    target_state, 
    t_array, 
    number_trajectories, 
    dt, 
    psi_0_j, 
    I_H_0_tilde_eff_j, 
    L_operators_j, 
    Ldag_operators_j, 
    sum_LdagL_operators_j, 
    I_H_c_tilde_real_j, 
    I_H_c_tilde_imag_j, 
    I_imag_j, 
    N_Hamiltonian_steps,
    initial_delta_p=None
):
    """
    Optimizes the relative variation parameters delta_p for a given pulse.
    
    Args:
        pulse_id (str): Type of pulse ('initialization', 'storage', 'retrieval').
        t_pulse_center (float): Center time of the pulse.
        omega_RF (float): Rotating frame frequency.
        target_state (jnp.ndarray): Target state in real representation.
        time_array (jnp.ndarray): Array of time values.
        number_trajectories (int): Number of trajectories for simulation.
        dt (float): Time step for simulation.
        psi_0_j (jnp.ndarray): Initial state in real representation.
        I_H_0_tilde_eff_j (jnp.ndarray): Effective Hamiltonian in rotating frame.
        L_operators_j (jnp.ndarray): Collapse operators.
        Ldag_operators_j (jnp.ndarray): Daggered collapse operators.
        sum_LdagL_operators_j (jnp.ndarray): Sum of Ldag*L operators.
        I_H_c_tilde_real_j (jnp.ndarray): Real part of control Hamiltonian.
        I_H_c_tilde_imag_j (jnp.ndarray): Imaginary part of control Hamiltonian.
        I_imag_j (jnp.ndarray): Imaginary unit matrix.
        N_Hamiltonian_steps (int): Number of Hamiltonian steps.
        initial_delta_p (list, optional): Initial guess for delta_p. Defaults to [0, 0, 0, 0].
    
    Returns:
        tuple: Optimized delta_p and the minimum cost.
    """
    if initial_delta_p is None:
        initial_delta_p = [0.0, 0.0, 0.0, 0.0]
    
    def cost_wrapper(delta_p):
        """Wrapper function to compute cost for given delta_p."""
        # Apply delta_p to the default pulse parameters
        default_pulse = get_default_pulse(pulse_id, t_pulse_center, omega_RF)
        
        # Extract default parameters
        hbar_omega_P_default = default_pulse.hbar_omega_P
        tau_0_P_default = default_pulse.tau_0_P
        GDD_P_default = default_pulse.GDD_P
        Theta_P_default = default_pulse.Theta_P
        
        # Apply relative variations
        delta_GDD_P, delta_Theta_P, delta_hbar_omega_P, delta_tau_0_P = delta_p
        hbar_omega_P = hbar_omega_P_default * (1 + delta_hbar_omega_P)
        tau_0_P = tau_0_P_default * (1 + delta_tau_0_P)
        GDD_P = GDD_P_default * (1 + delta_GDD_P)
        Theta_P = Theta_P_default * (1 + delta_Theta_P)
        
        # Create a new pulse with modified parameters
        modified_pulse = get_default_pulse(pulse_id, t_pulse_center, omega_RF)
        modified_pulse.hbar_omega_P = hbar_omega_P
        modified_pulse.tau_0_P = tau_0_P
        modified_pulse.GDD_P = GDD_P
        modified_pulse.Theta_P = Theta_P

        jdebug.breakpoint()
        
        # Get the modified pulse function
        modified_pulse_function = modified_pulse.get_chirped_pulse_function()
        
        # Create input array with the modified pulse
        input_array = jnp.array(modified_pulse_function(t_array))
        
        # Simulate the system with the modified pulse
        traj = simulate_batch(
            random_key=42,  # Fixed random key for reproducibility
            input_array=input_array,
            number_trajectories=number_trajectories,
            number_steps=len(t_array),
            dt=dt,
            psi_0_j=psi_0_j,
            I_H_0_tilde_eff_j=I_H_0_tilde_eff_j,
            L_operators_j=L_operators_j,
            Ldag_operators_j=Ldag_operators_j,
            sum_LdagL_operators_j=sum_LdagL_operators_j,
            I_H_c_tilde_real_j=I_H_c_tilde_real_j,
            I_H_c_tilde_imag_j=I_H_c_tilde_imag_j,
            dt_sim=dt,
            I_imag_j=I_imag_j,
            N_Hamiltonian_steps=N_Hamiltonian_steps
        )
        
        # Compute the cost
        cost = cost_function_target_state(traj, target_state, t_array)
        return cost
    
    # Optimize delta_p using scipy.optimize.minimize
    result = ScipyBoundedMinimize(fun=cost_wrapper, method='SLSQP')
    result_params = result.run(init_params=initial_delta_p, bounds=([-1e-3, -0.2, -0.2, -0.2], [1e-3, 0.2, 0.2, 0.2])).params
    
    return result_params


def main():
    # Load a timer
    timer = Timer()
    # Load QD parameters
    par_QD = get_parameters_QD()
    # Choose initial and target state (G, X_H, X_V, D_H, D_V, B)
    init_state_sim = "G"
    # no. of Trajectories & sample times for optimization & simulation
    number_trajectories_opt = 30
    number_trajectories_sim = 30
    dt_opt = 0.2e-2* par_QD.hbar/par_QD.E_B* 2*jnp.pi
    dt_sim = 0.2e-2* par_QD.hbar/par_QD.E_B* 2*jnp.pi
    N_Hamiltonian_steps = 2
    # Polarization overlaps (e_H * e_L, e_V * e_L)
    polarization_overlaps = jnp.array([1,0])
    omega_X = par_QD.hbar_omega_X/par_QD.hbar

    # initialize the simulations (time_array actually only for the final overall sim)
    time_array_sim = jnp.arange(0, 1520, dt_sim)
    time_array_opt = jnp.arange(0, 1520, dt_opt)
    number_time_steps_opt = len(time_array_opt)
    number_time_steps_sim = len(time_array_sim)
    H_0_tilde_eff_j, L_operators_j, Ldag_operators_j, sum_LdagL_operators_j, psi_0_j,  H_c_tilde_real_j, H_c_tilde_imag_j, I_imag_j, I_H_0_tilde_eff_j, I_H_c_tilde_real_j, I_H_c_tilde_imag_j, U_rotating_frame_j = jax_sim_setup(init_state_sim, polarization_overlaps, time_array_sim, par_QD)
    
    # obtain default pulses and their parameters
    time_until_init_pulse = 30 #ps
    init_pulse_default = get_default_pulse("initialization", time_until_init_pulse + 0, omega_RF=omega_X)
    init_pulse_function = init_pulse_default.get_chirped_pulse_function()
    # init_pulse_parameters = init_pulse_default.get_pulse_parameters_dict()
    storage_pulse_default = get_default_pulse("storage", time_until_init_pulse + 70, omega_RF=omega_X)
    storage_pulse_function = storage_pulse_default.get_chirped_pulse_function()
    # storage_pulse_parameters = storage_pulse_default.get_pulse_parameters_dict()
    retrieval_pulse_default = get_default_pulse("retrieval", time_until_init_pulse + 1420, omega_RF=omega_X)
    retrieval_pulse_function = retrieval_pulse_default.get_chirped_pulse_function()
    # retrieval_pulse_parameters = retrieval_pulse_default.get_pulse_parameters_dict()

    # create target states
    target_init = complex_to_real_vector(create_target_state("B"))
    target_storage = complex_to_real_vector(create_target_state("D_H"))
    target_retrieval = complex_to_real_vector(create_target_state("B"))


    # optimize the init pulse
    delta_p_init = optimize_delta_p("initialization", time_until_init_pulse + 0, omega_X, target_init, time_array_opt, number_trajectories_opt, dt_opt, psi_0_j, I_H_0_tilde_eff_j, L_operators_j, Ldag_operators_j, sum_LdagL_operators_j, I_H_c_tilde_real_j, I_H_c_tilde_imag_j, I_imag_j, N_Hamiltonian_steps,initial_delta_p=None)

    print(delta_p_init)
    
    # define the baseline input array
    # hard factor in front of input
    input_factor = 1
    input_array = input_factor * jnp.array( init_pulse_function(time_array_sim) + storage_pulse_function(time_array_sim) + retrieval_pulse_function(time_array_sim) )

    # simulate the system
    timer.start()
    traj_sim = np.array(simulate_batch(200, input_array, number_trajectories_sim, number_time_steps_sim, dt_sim, psi_0_j, I_H_0_tilde_eff_j, L_operators_j, Ldag_operators_j, sum_LdagL_operators_j, I_H_c_tilde_real_j, I_H_c_tilde_imag_j, dt_sim, I_imag_j, N_Hamiltonian_steps) )
    _ = timer.stop()
    
    # Create QDSimResults object and plot results
    sim_results = QDSimResults(np.array(traj_sim), time_array_sim, par_QD, input_array)
    sim_results.plot_control_field_real("")
    sim_results.plot_mean_population_trajectories()
    sim_results.plot_mean_dressed_population_trajectories()


if __name__ == "__main__":

    main()

