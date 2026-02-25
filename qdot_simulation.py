# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, vmap, random # , profiler
from jax.lax import scan, fori_loop
from getparameters import ParametersQD
import warnings

from utilities import *

def create_QD_hamiltonian_terms_and_states(params_QD: ParametersQD):
    """Create the Hamiltonian terms and state vectors for a quantum dot system.

    Args:
        params_QD (object): An object containing the following attributes:
            E_X_H (float): Exciton horizontal energy
            E_X_V (float): Exciton vertical energy
            E_D_H (float): Dark exciton horizontal energy
            E_D_V (float): Dark exciton vertical energy
            E_XX (float): Biexciton energy
            mu_B (float): Bohr magneton
            B_x (float): Magnetic field in x direction
            B_z (float): Magnetic field in z direction
            g_ex (float): Electron g factor
            g_hx (float): Hole g factor

    Returns:
        tuple: A tuple containing the state vectors and the Hamiltonian matrix.
            state_vectors (tuple): A tuple of state vectors for the ground state, exciton_H, exciton_V, dark_exciton_H, dark_exciton_V, and biexciton.
            H_0 (numpy.ndarray): The Hamiltonian matrix for the system.
    """
    E_X_H = params_QD.E_X_H  # Exciton horizontal energy
    E_X_V = params_QD.E_X_V  # Exciton vertical energy
    E_D_H = params_QD.E_D_H  # Dark exciton horizontal energy
    E_D_V = params_QD.E_D_V  # Dark exciton vertical energy
    E_XX = params_QD.E_XX     # Biexciton energy
    mu_b = params_QD.mu_B    # Bohr magneton
    bx = params_QD.B_x       # Magnetic field in x direction
    bz = params_QD.B_z       # Magnetic field in z direction
    g_ex = params_QD.g_ex    # Electron g factor
    g_hx = params_QD.g_hx    # Hole g factor
    g_ez = 0              # Placeholder for electron g factor in z direction
    g_hz = 0              # Placeholder for hole g factor in z direction

    # Define the six-level system
    ground_state = jnp.zeros((6,1), dtype=jnp.complex64)
    ground_state = ground_state.at[0].set(1)
    exciton_H = jnp.zeros((6,1), dtype=jnp.complex64)
    exciton_H = exciton_H.at[1].set(1)
    exciton_V = jnp.zeros((6,1), dtype=jnp.complex64)
    exciton_V = exciton_V.at[2].set(1)
    dark_exciton_H = jnp.zeros((6,1), dtype=jnp.complex64)
    dark_exciton_H = dark_exciton_H.at[3].set(1)
    dark_exciton_V = jnp.zeros((6,1), dtype=jnp.complex64)
    dark_exciton_V = dark_exciton_V.at[4].set(1)
    biexciton = jnp.zeros((6,1), dtype=jnp.complex64)
    biexciton = biexciton.at[5].set(1)

    # Define the Hamiltonian terms
    H_QD = E_X_H * exciton_H @ exciton_H.conj().T + E_X_V * exciton_V @ exciton_V.conj().T + \
         E_D_H * dark_exciton_H @ dark_exciton_H.conj().T + E_D_V * dark_exciton_V @ dark_exciton_V.conj().T + \
         E_XX * biexciton @ biexciton.conj().T

    # Bright-dark coupling depending on Bx
    if bx != 0:
        H_bx = -0.5 * mu_b * bx * (g_hx + g_ex) * (exciton_H @ dark_exciton_H.conj().T + dark_exciton_H @ exciton_H.conj().T) + \
              -0.5 * mu_b * bx * (g_hx - g_ex) * (exciton_V @ dark_exciton_V.conj().T + dark_exciton_V @ exciton_V.conj().T)
    else:
        H_bx = jnp.zeros((6, 6))

    # Bright-bright and dark-dark coupling depending on Bz
    if bz != 0.0:
        H_bz = 1j * 0.5 * mu_b * bz * (g_ez - 3 * g_hz) * (exciton_H @ exciton_V.conj().T - exciton_V @ exciton_H.conj().T) + \
              1j * -0.5 * mu_b * bz * (g_ez + 3 * g_hz) * (dark_exciton_H @ dark_exciton_V.conj().T - dark_exciton_V @ dark_exciton_H.conj().T)
    else:
        H_bz = jnp.zeros((6, 6))

    # prepare output
    state_vectors = ground_state, exciton_H, exciton_V, dark_exciton_H, dark_exciton_V, biexciton
    H_0 = H_QD + H_bx + H_bz
    
    return state_vectors, H_0


def create_dressed_energies_and_states(params_QD: ParametersQD):
    """create the eigenenergies and eigenstates of the dressed state system.

    Args:
        params_QD (object): An object containing the following parameters:
            E_X_H (float): Exciton horizontal energy
            E_X_V (float): Exciton vertical energy
            E_D_H (float): Dark exciton horizontal energy
            E_D_V (float): Dark exciton vertical energy
            E_XX (float): Biexciton energy
            mu_B (float): Bohr magneton
            B_x (float): Magnetic field in x direction
            g_ex (float): Electron g factor
            g_hx (float): Hole g factor

    Returns:
        tuple: A tuple containing:
            - dressed_energies (tuple): A tuple of dressed state energies:
                - E_G_dressed (float): Dressed ground state energy
                - E_X_H_dressed (float): Dressed exciton horizontal energy
                - E_X_V_dressed (float): Dressed exciton vertical energy
                - E_D_H_dressed (float): Dressed dark exciton horizontal energy
                - E_D_V_dressed (float): Dressed dark exciton vertical energy
                - E_B_dressed (float): Dressed biexciton energy
            - dressed_states (tuple): A tuple of dressed state vectors:
                - ground_state_dressed (array): Dressed ground state vector
                - exciton_H_dressed (array): Dressed exciton horizontal state vector
                - exciton_V_dressed (array): Dressed exciton vertical state vector
                - dark_exciton_H_dressed (array): Dressed dark exciton horizontal state vector
                - dark_exciton_V_dressed (array): Dressed dark exciton vertical state vector
                - biexciton_dressed (array): Dressed biexciton state vector
            - dressed_states_transform (array): A matrix formed by concatenating the dressed state vectors horizontally
    """
    # parameters
    E_G = 0               # Ground state energy (0 by definition)
    E_X_H = params_QD.E_X_H  # Exciton horizontal energy
    E_X_V = params_QD.E_X_V  # Exciton vertical energy
    E_D_H = params_QD.E_D_H  # Dark exciton horizontal energy
    E_D_V = params_QD.E_D_V  # Dark exciton vertical energy
    E_B = params_QD.E_XX     # Biexciton energy
    mu_b = params_QD.mu_B    # Bohr magneton
    bx = params_QD.B_x       # Magnetic field in x direction
    g_ex = params_QD.g_ex    # Electron g factor
    g_hx = params_QD.g_hx    # Hole g factor
    # abbreviation for the Hamiltonian
    j_plus =  -0.5 * mu_b * bx * (g_hx + g_ex)
    j_minus =  -0.5 * mu_b * bx * (g_hx - g_ex)
    # avoid division by
    # define new eigenenergies and eigenvectors (dressed states)
    state_vectors, _ = create_QD_hamiltonian_terms_and_states(params_QD)
    ground_state, exciton_H, exciton_V, dark_exciton_H, dark_exciton_V, biexciton = state_vectors
    E_G_dressed = E_G #stays the same
    E_X_H_dressed = 0.5* ( jnp.sqrt( (E_X_H-E_D_H)**2 + 4*j_plus**2 ) + E_X_H + E_D_H )
    E_D_H_dressed = 0.5* (-jnp.sqrt( (E_X_H-E_D_H)**2 + 4*j_plus**2 ) + E_X_H + E_D_H )
    if jnp.abs(j_minus) < 1e-10:
        E_X_V_dressed = E_X_V
        E_D_V_dressed = E_D_V
    else:
        E_X_V_dressed = 0.5* ( jnp.sqrt( (E_X_V-E_D_V)**2 + 4*j_minus**2 ) + E_X_V + E_D_V )
        E_D_V_dressed = 0.5* (-jnp.sqrt( (E_X_V-E_D_V)**2 + 4*j_minus**2 ) + E_X_V + E_D_V )
    E_B_dressed = E_B #stays the same
    ground_state_dressed = ground_state # stays the same
    exciton_H_dressed = normalize_psi(
        (jnp.sqrt( (E_X_H-E_D_H)**2 + 4*j_plus**2 ) + E_X_H - E_D_H)/(2*j_plus)*exciton_H + dark_exciton_H )
    if jnp.abs(j_minus) < 1e-10:
        exciton_V_dressed = exciton_V
    else:
        exciton_V_dressed = normalize_psi(
            (jnp.sqrt( (E_X_V-E_D_V)**2 + 4*j_minus**2 ) + E_X_V - E_D_V)/(2*j_minus)*exciton_V + dark_exciton_V )
    dark_exciton_H_dressed = normalize_psi(
        (-jnp.sqrt( (E_X_H-E_D_H)**2 + 4*j_plus**2 ) + E_X_H - E_D_H)/(2*j_plus)*exciton_H + dark_exciton_H )
    if jnp.abs(j_minus) < 1e-10:
        dark_exciton_V_dressed = dark_exciton_V
    else:
        dark_exciton_V_dressed = normalize_psi(
            (-jnp.sqrt( (E_X_V-E_D_V)**2 + 4*j_minus**2 ) + E_X_V - E_D_V)/(2*j_minus)*exciton_V + dark_exciton_V )
    # exciton_H_dressed = (
    #     (jnp.sqrt( (E_X_H-E_D_H)**2 + 4*j_plus**2 ) + E_X_H - E_D_H)/(2*j_plus)*exciton_H + dark_exciton_H )
    # if jnp.abs(j_minus) < 1e-10:
    #     exciton_V_dressed = exciton_V
    # else:
    #     exciton_V_dressed = (
    #         (jnp.sqrt( (E_X_V-E_D_V)**2 + 4*j_minus**2 ) + E_X_V - E_D_V)/(2*j_minus)*exciton_V + dark_exciton_V )
    # dark_exciton_H_dressed = (
    #     (-jnp.sqrt( (E_X_H-E_D_H)**2 + 4*j_plus**2 ) + E_X_H - E_D_H)/(2*j_plus)*exciton_H + dark_exciton_H )
    # if jnp.abs(j_minus) < 1e-10:
    #     dark_exciton_V_dressed = dark_exciton_V
    # else:
    #     dark_exciton_V_dressed = (
    #         (-jnp.sqrt( (E_X_V-E_D_V)**2 + 4*j_minus**2 ) + E_X_V - E_D_V)/(2*j_minus)*exciton_V + dark_exciton_V )
    biexciton_dressed = biexciton #stays the same

    dressed_energies = E_G_dressed, E_X_H_dressed, E_X_V_dressed, E_D_H_dressed, E_D_V_dressed, E_B_dressed
    dressed_states = ground_state_dressed, exciton_H_dressed, exciton_V_dressed, dark_exciton_H_dressed, dark_exciton_V_dressed, biexciton_dressed
    # Create the dressed_states_transform matrix by concatenating/stacking dressed states horizontally
    dressed_states_transform = jnp.concatenate(dressed_states, axis=1)

    return dressed_energies, dressed_states, dressed_states_transform


def transform_Hamiltonian_to_rotating_frame(H_0, params_QD: ParametersQD, t_array):
    """
    Transforms the Hamiltonian to the rotating frame and computes the rotating frame unitary.

    Args:
        H_0 (jnp.ndarray): The Hamiltonian in the original frame.
        t_array (jnp.ndarray): Array of time values.
        params_QD: An object containing the parameters hbar_omega_X and hbar.

    Returns:
        tuple: A tuple containing:
            - H_0_tilde_eff (jnp.ndarray): The effective Hamiltonian in the rotating frame.
            - omega_RF (float): The rotating frame frequency.
            - U_rotating_frame (jnp.ndarray): The rotating frame unitary for each time in t_array.
    """
    hbar_omega_X = params_QD.hbar_omega_X
    omega_RF = hbar_omega_X / params_QD.hbar
    rotating_frame_diagonal = jnp.array([0, 1, 1, 1, 1, 2])

    # Transform H_0 to rotating frame and subtract derivative of U to gain effective H_0_tilde (see Maple)
    H_0_tilde_eff = H_0 - hbar_omega_X * jnp.diag(rotating_frame_diagonal)

    mu_b = params_QD.mu_B    # Bohr magneton
    bx = params_QD.B_x       # Magnetic field in x direction
    g_ex = params_QD.g_ex    # Electron g factor
    g_hx = params_QD.g_hx    # Hole g factor
    # abbreviation for the Hamiltonian
    j_plus =  -0.5 * mu_b * bx * (g_hx + g_ex)
    j_minus =  -0.5 * mu_b * bx * (g_hx - g_ex)

    # H_0_tilde_eff = jnp.array([[0,0,0,0,0,0], [0,0.5*params_QD.delta_X,0,j_plus,0,0], [0,0,-0.5*params_QD.delta_X,0,j_minus,0], [0,j_plus,0,-0.1+0.5*params_QD.delta_D,0,0], [0,0,j_minus,0,-0.1-0.5*params_QD.delta_D,0], [0,0,0,0,0,-params_QD.E_B]]) # !!! DEBUG: this was just introduced due to the errors in the above computation of H_0_tilde_eff. This, however, is just hard-coded (not ideal!)

    # Define function of the transform
    U_rotating_frame_fun = lambda t: jnp.diag(jnp.exp(-1j * omega_RF * t * rotating_frame_diagonal))
    U_rotating_frame_vmap = vmap(U_rotating_frame_fun, in_axes=0, out_axes=0)
    U_rotating_frame = U_rotating_frame_vmap(t_array)

    return H_0_tilde_eff, omega_RF, U_rotating_frame
    

def create_collapse_operators(state_vectors, params_QD: ParametersQD):
    """
    Create collapse operators and their squares for a given set of state vectors and parameters.

    Parameters:
    state_vectors (tuple): A tuple of state vectors (ground_state, exciton_H, exciton_V, _, _, biexciton).
    params_QD (object): An object containing parameters such as Gamma_X_inv and Gamma_XX_inv.

    Returns:
    tuple: A tuple containing two lists:
        - collapse_ops (list): List of collapse operators.
        - squared_collapse_ops (list): List of squares of collapse operators (Ldag*L).
    """
    # Unpack the state vectors
    ground_state, exciton_H, exciton_V, _, _, biexciton = state_vectors

    # Define decay rates for each state
    exciton_decay_rate = 1 / params_QD.Gamma_X_inv  # Exciton decay rate
    # dark_state_decay_rate = 0  # Decay rate for dark states
    biexciton_decay_rate = 1 / params_QD.Gamma_XX_inv  # Biexciton decay rate

    # Create collapse operators with their respective decay rates
    collapse_ops = [
        jnp.sqrt(exciton_decay_rate) * (ground_state @ exciton_H.conj().T),
        jnp.sqrt(exciton_decay_rate) * (ground_state @ exciton_V.conj().T),
        # # Leave out the dark states to reduce overhead
        # np.sqrt(dark_state_decay_rate) * (ground_state @ dark_exciton_x.conj().T),
        # np.sqrt(dark_state_decay_rate) * (ground_state @ dark_exciton_y.conj().T),
        jnp.sqrt(biexciton_decay_rate) * (exciton_H @ biexciton.conj().T),
        jnp.sqrt(biexciton_decay_rate) * (exciton_V @ biexciton.conj().T)
    ]

    # Compute the various L_dag * L
    squared_collapse_ops = [L.conj().T @ L for L in collapse_ops]

    return collapse_ops, squared_collapse_ops


def create_control_Hamiltonians_rotating(pol_overlaps, state_vectors):
    """
    Create control Hamiltonians for rotating frame.

    Args:
        omega_RF: RF frequency
        omega_p: Input pulse carrier frequency
        t_array: Time array
        pol_overlaps: Polarization overlaps
        state_vectors: State vectors

    Returns:
        H_c_tilde_real: part of the control Hamiltonian that multiplies with real(Omega)
        H_c_tilde_imag: part of the control Hamiltonian that multiplies with imag(Omega)
    """
    ground_state, exciton_H, exciton_V, dark_exciton_H, dark_exciton_V, biexciton = state_vectors
    H_c_1 = pol_overlaps[0]*( exciton_H @ ground_state.conj().T + biexciton @ exciton_H.conj().T ) + pol_overlaps[1]*( exciton_V @ ground_state.conj().T + biexciton @ exciton_V.conj().T )
    H_c_2 = pol_overlaps[0].conj()*( ground_state @ exciton_H.conj().T + exciton_H @ biexciton.conj().T ) + pol_overlaps[1].conj()*( ground_state @ exciton_V.conj().T + exciton_V @ biexciton.conj().T )

    # H_c_tilde_real_fun = lambda t: jnp.exp(1j*(omega_RF-omega_p)*t)* H_c_1 + jnp.exp(-1j*(omega_RF-omega_p)*t)* H_c_2
    # H_c_tilde_imag_fun = lambda t: 1j* jnp.exp(1j*(omega_RF-omega_p)*t)* H_c_1 -1j* jnp.exp(-1j*(omega_RF-omega_p)*t)* H_c_2
    # H_c_tilde_real_vmap = vmap(H_c_tilde_real_fun, in_axes=0, out_axes=0)
    # H_c_tilde_imag_vmap = vmap(H_c_tilde_imag_fun, in_axes=0, out_axes=0)

    # #evaluate the vmaps to obtain the H_c for the real and imag parts of Omega_tilde
    # H_c_tilde_real = H_c_tilde_real_vmap(t_array)
    # H_c_tilde_imag = H_c_tilde_imag_vmap(t_array)

    H_c_tilde_real = H_c_1 + H_c_2
    H_c_tilde_imag = 1j*H_c_1 - 1j*H_c_2

    return H_c_tilde_real, H_c_tilde_imag


def create_initial_state(psi_0_choice: str, state_vectors) -> jnp.ndarray:
    """
    Create the initial state for the system.

    Args:
        psi_0_choice: A string representing the initial state choice.
        state_vectors: A tuple of numpy arrays representing the state vectors.

    Returns:
        A jax.numpy array representing the initial state.

    Raises:
        ValueError: If the psi_0_choice is not a valid key in state_map.
    """
    # Unpack the state vectors
    ground_state, exciton_H, exciton_V, dark_exciton_H, dark_exciton_V, biexciton = state_vectors

    # Create a mapping from state names to state vectors
    state_map = {
        "G": ground_state,
        "X_H": exciton_H,
        "X_V": exciton_V,
        "D_H": dark_exciton_H,
        "D_V": dark_exciton_V,
        "B": biexciton
    }

    # Get the initial state based on the choice
    psi_0 = state_map.get(psi_0_choice, None)

    if psi_0 is None:
        warnings.warn(f"Invalid initial state choice '{psi_0_choice}'. Defaulting to ground_state.")
        psi_0 = ground_state

    print("The chosen initial state is: ", psi_0_choice)

    return psi_0


def create_target_state(psi_T_choice: str, state_vectors) -> jnp.ndarray:
    """
    Create the target state for the system.

    Args:
        psi_T_choice: A string representing the target state choice.
        state_vectors: A tuple of numpy arrays representing the state vectors.

    Returns:
        A jax.numpy array representing the target state.

    Raises:
        ValueError: If the psi_T_choice is not a valid key in state_map.
    """
    # Unpack the state vectors
    ground_state, exciton_H, exciton_V, dark_exciton_H, dark_exciton_V, biexciton = state_vectors

    # Create a mapping from state names to state vectors
    state_map = {
        "G": ground_state,
        "X_H": exciton_H,
        "X_V": exciton_V,
        "D_H": dark_exciton_H,
        "D_V": dark_exciton_V,
        "B": biexciton
    }

    # Get the initial state based on the choice
    psi_T = state_map.get(psi_T_choice, None)

    if psi_T is None:
        warnings.warn(f"Invalid target state choice '{psi_T_choice}'. Defaulting to ground_state.")
        psi_T = ground_state

    print("The chosen target state is: ", psi_T_choice)

    return psi_T


def jax_sim_setup(psi_0_choice, psi_T_choice, pol_overlaps, t_array, params_QD: ParametersQD):
    """
    Sets up the simulation for the quantum dot system in the rotating frame.

    Args:
        psi_0_choice (str): Choice for the initial state.
        psi_T_choice (str): Choice for the target state.
        params_QD (dict): Parameters for the quantum dot system.

    Returns:
        tuple: A tuple containing the following elements:
            - H_0_tilde_eff_j (jnp.ndarray): Effective Hamiltonian in the rotating frame.
            - L_operators_j (jnp.ndarray): Collapse operators.
            - LdagL_operators_j (jnp.ndarray): Products of collapse operators and their conjugates.
            - psi_0_j (jnp.ndarray): Initial state vector.
            - psi_T_j (jnp.ndarray): Target state vector.
            - H_c_tilde_real_j (jnp.ndarray): Real part of the control Hamiltonian in the rotating frame.
            - H_c_tilde_imag_j (jnp.ndarray): Imaginary part of the control Hamiltonian in the rotating frame.
            - I_imag_j (jnp.ndarray): Imaginary unit matrix.
    """
    # Create state vectors and nominal Hamiltonian
    state_vectors, H_0 = create_QD_hamiltonian_terms_and_states(params_QD)
    # transform Hamiltonian to the rotating frame (that rotates at omega_X)
    H_0_tilde_eff, omega_RF, U_rotating_frame = transform_Hamiltonian_to_rotating_frame(H_0, params_QD, t_array)
    # Create collapse operators
    L_operators, LdagL_operators = create_collapse_operators(state_vectors, params_QD)
    Ldag_operators = [L.conj().T for L in L_operators]
    # Create initial state
    psi_0 = create_initial_state(psi_0_choice, state_vectors)
    # Create target state
    psi_T = create_target_state(psi_T_choice, state_vectors)
    # create the control Hamiltonians in rotating frame
    H_c_tilde_real, H_c_tilde_imag = create_control_Hamiltonians_rotating(pol_overlaps, state_vectors)

    # convert to real JAX matrices / vectors
    H_0_tilde_eff_j = jnp.array( complex_to_real_block(H_0_tilde_eff) )
    L_operators_j = jnp.stack( [jnp.array(complex_to_real_block(L)) for L in L_operators] )
    Ldag_operators_j = jnp.stack( [jnp.array(complex_to_real_block(L)) for L in Ldag_operators] )
    LdagL_operators_j = jnp.stack( [jnp.array(complex_to_real_block(LdagL)) for LdagL in LdagL_operators] )
    sum_LdagL_operators_j = jnp.sum(LdagL_operators_j, axis=0)
    psi_0_j = jnp.concatenate([psi_0.real, psi_0.imag])
    psi_T_j = jnp.concatenate([psi_T.real, psi_T.imag])
    H_c_tilde_real_j = jnp.array( complex_to_real_block(H_c_tilde_real) )
    H_c_tilde_imag_j = jnp.array( complex_to_real_block(H_c_tilde_imag) )
    I_imag_j = jnp.block([[jnp.zeros((6, 6)), -jnp.eye(6)], [jnp.eye(6), jnp.zeros((6, 6))]])
    complex_to_real_block_vmap = vmap( complex_to_real_block, in_axes=0, out_axes=0 )
    U_rotating_frame_j = complex_to_real_block_vmap( U_rotating_frame )

    #pre-compute the Hamiltonians times the complex unit (divided by hbar)
    I_H_0_tilde_eff_j = I_imag_j @ H_0_tilde_eff_j / params_QD.hbar
    I_H_c_tilde_real_j = I_imag_j @ H_c_tilde_real_j
    I_H_c_tilde_imag_j = I_imag_j @ H_c_tilde_imag_j

    return H_0_tilde_eff_j, L_operators_j, Ldag_operators_j, sum_LdagL_operators_j, psi_0_j, psi_T_j, H_c_tilde_real_j, H_c_tilde_imag_j, I_imag_j, I_H_0_tilde_eff_j, I_H_c_tilde_real_j, I_H_c_tilde_imag_j, U_rotating_frame_j


def create_jax_noise_traj_arrays(key_index: int, number_collapse: int, number_trajectories: int, number_steps: int, dt: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate JAX noise trajectory arrays for quantum trajectories simulation.

    Args:
        key_index (int): which key is used for RNG.
        number_collapse (int): Number of collapse operators.
        number_trajectories (int): Number of trajectories to simulate.
        number_steps (int): Number of time steps in each trajectory.
        dt (float): Time step size.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: A tuple of two JAX arrays representing the real and imaginary parts of the noise trajectories.
    """
    # Create a JAX random key from a fixed seed
    key = random.PRNGKey(key_index)

    # Generate noise arrays using JAX random functions
    dW_real_j = random.normal(key, (number_trajectories, number_steps, number_collapse)) * jnp.sqrt(dt / 2)
    key, subkey = random.split(key)
    dW_imag_j = random.normal(subkey, (number_trajectories, number_steps, number_collapse)) * jnp.sqrt(dt / 2)

    return dW_real_j, dW_imag_j


@jit
def normalize_psi(psi):
    regularization =   1e-14 # prevent division by zero
    norm = jnp.linalg.norm(psi)
    return psi / (norm + regularization)


#@jit
def em_step(psi_in, Omega_real, Omega_imag, dW_real, dW_imag, I_H_0_tilde_eff, L_operators, Ldag_operators, sum_LdagL_operators, I_H_c_tilde_real, I_H_c_tilde_imag, dt, I_imag, N_Hamilton_steps=1):
    """
    Perform an Euler-Maruyama step for the quantum dot system.
    
    Args:
        psi: Current state vector (real block representation)
        I_H_0_tilde_eff: Effective Hamiltonian in rotating frame (real block representation)
        L_operators: Collapse operators (real block representation)
        Ldag_operators: Daggered collapse operators (real block representation)
        sum_LdagL_operators: sum over the Ldag*L operators (real block representation)
        I_H_c_tilde_real: Real part of control Hamiltonian (real block representation)
        I_H_c_tilde_imag: Imaginary part of control Hamiltonian (real block representation)
        Omega_real: Real part of control field
        Omega_imag: Imaginary part of control field
        dt: Time step
        dW_real: Real part of Wiener process increments
        dW_imag: Imaginary part of Wiener process increments
        N_Hamilton_steps: Number of RK2 steps to perform for the Hamiltonian term
        
    Returns:
        Updated state vector after one Euler-Maruyama step
    """
    
    def hamiltonian_rk2_step(psi, I_H_total, dt_step):
        """Perform a single RK2 step for the Hamiltonian evolution."""
        # RK2 step 1
        k1 = -dt_step * I_H_total @ psi
        psi_mid = psi + 0.5 * k1
        
        # RK2 step 2
        k2 = -dt_step * I_H_total @ psi_mid
        psi_new = psi + k2
        
        return psi_new
    
    # Compute the total Hamiltonian
    I_H_total = I_H_0_tilde_eff + Omega_real * I_H_c_tilde_real + Omega_imag * I_H_c_tilde_imag
    
    # Perform N_Hamilton_steps RK2 steps for the Hamiltonian term
    # Use fori_loop to apply the RK2 steps sequentially
    dt_step = dt / N_Hamilton_steps  # Sub-step size
    def body_fn(i, psi):
        return hamiltonian_rk2_step(psi, I_H_total, dt_step)
    
    psi = normalize_psi(fori_loop(0, N_Hamilton_steps, body_fn, psi_in))

    # compute the state average of the Ldag operators
    dim_half = I_H_0_tilde_eff.shape[1] // 2
    psidag = psi.reshape(1, -1, 1)#.at[:,dim_half:].multiply(-1)
    Ldag_operators_avg = jnp.sum(psidag * (Ldag_operators @ psi), axis=1, keepdims=True)
    L_operators_avg = jnp.sum(psidag * (L_operators @ psi), axis=1, keepdims=True)

    # compute the L*psi and L_avg*psi that are used twice
    L_operators_psi = L_operators @ psi
    L_operators_avg_psi = L_operators_avg*psi
    
    # Deterministic drift term
    drift = - 0.5*sum_LdagL_operators @ psi + jnp.sum( Ldag_operators_avg*L_operators_psi, axis=0 ) - 0.5*jnp.sum( Ldag_operators_avg*L_operators_avg_psi, axis=0 )

    # Stochastic diffusion terms
    dW_real_broadcast = dW_real.reshape(-1, 1, 1)  # Shape: (N_collapse, 1, 1)
    dW_imag_broadcast = dW_imag.reshape(-1, 1, 1)  # Shape: (N_collapse, 1, 1)    

    stochastic_terms = jnp.sum((L_operators_psi-L_operators_avg_psi)*dW_real_broadcast + I_imag @ (L_operators_psi-L_operators_avg_psi)*dW_imag_broadcast, axis=0)
    
    # Euler-Maruyama update
    psi_new = psi + dt * drift + stochastic_terms

    return normalize_psi(psi_new)


@jit
def simulate_single_traj(
        Omega_real_traj, Omega_imag_traj, dW_real_traj, dW_imag_traj, psi_0_j, I_H_0_tilde_eff, L_operators, Ldag_operators, sum_LdagL_operators, I_H_c_tilde_real, I_H_c_tilde_imag, dt, I_imag, N_Hamilton_steps=1):
    """
    Optimized version that avoids passing large time-dependent operators through scan.
    Instead, we use the time index to access operators from precomputed arrays.
    """
    
    def step_fn(carry, inputs):
        psi = carry
        dW_r, dW_i, u_r, u_i = inputs        
        psi_next = em_step(psi, u_r, u_i, dW_r, dW_i, I_H_0_tilde_eff, L_operators, Ldag_operators, sum_LdagL_operators, I_H_c_tilde_real, I_H_c_tilde_imag, dt, I_imag, N_Hamilton_steps)
        return psi_next, psi_next

    inputs = (dW_real_traj, dW_imag_traj, Omega_real_traj, Omega_imag_traj)
    _, traj = scan(step_fn, psi_0_j, inputs)
    
    return traj

sim_forward_vmap = vmap(simulate_single_traj, in_axes=(None, None, 0, 0, None, None, None, None, None, None, None, None, None, None), out_axes=0)


class QDSimResults:
    """Class to hold quantum dot simulation results and provide plotting methods."""
    
    def __init__(self, trajs, t_array, params_QD, control_FF):
        """Initialize QDSimResults with simulation data.
        
        Args:
            trajs: Trajectories from quantum dot simulation
            t_array: Time array for the simulation
            params_QD: ParametersQD object containing quantum dot parameters
            control_FF: Control field function or array
        """
        self.trajs = trajs
        self.t_array = t_array
        self.params_QD = params_QD
        self.control_FF = control_FF
    
    def plot_population_trajectories(self):
        """Plot population trajectories for all states (mean) and multiple (randomly chosen) trajectories."""
        # Convert to numpy array once
        all_trajs_np = np.array(self.trajs)
        # Compute all populations at once using numpy operations
        real_parts = all_trajs_np[:, :, :6, 0]  # Shape: (n_trajs, n_times, 6)
        imag_parts = all_trajs_np[:, :, 6:2*6, 0]  # Shape: (n_trajs, n_times, 6)
        # Compute populations: |ψ|² = Re² + Im²
        populations = real_parts**2 + imag_parts**2  # Shape: (n_trajs, n_times, 6)
        # Vectorized mean calculation
        mean_populations = np.mean(populations, axis=0)  # Shape: (n_times, 6)
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # Plot mean population trajectories
        state_labels = ["|G>", "|X_H>", "|X_V>", "|D_H>", "|D_V>", "|B>"]
        for i in range(6):
            ax1.plot(self.t_array, mean_populations[:, i], 
                    label=f"|{i}> = {state_labels[i]}")
        ax1.set_xlabel("Time (ps)")
        ax1.set_ylabel("Population")
        ax1.set_title("Mean Population Trajectories")
        ax1.legend(loc = "right")
        ax1.grid()

        # Randomly select 20 trajectories to plot
        n_selected = min(10, all_trajs_np.shape[0])  # Handle case with fewer than 10 trajectories
        selected_indices = np.random.choice(all_trajs_np.shape[0], size=n_selected, replace=False)
        # Plot individual trajectories more efficiently
        selected_populations = populations[selected_indices, :, :]  # Shape: (n_selected, n_times, 6)
        
        cmap = plt.get_cmap('tab20')

        for i in range(6):
            # Plota all selected trajectories for this state at once
            for j in range(n_selected):
                ax2.plot(self.t_array, selected_populations[j, :, i],
                         alpha=0.5, color=cmap(i * 2))
                # Add label for the state
            ax2.plot([], [], color=cmap(i * 2), label=state_labels[i], alpha=0.5)
        ax2.set_xlabel("Time (ps)")
        ax2.set_ylabel("Population")
        ax2.set_title("Individual Population Trajectories")
        ax2.legend(loc = "right")
        ax2.grid()
        plt.tight_layout()
        plt.show()
    
    def plot_mean_population_trajectories(self):
        """Plot mean population trajectories for all states."""
        # Convert to numpy array once
        all_trajs_np = np.array(self.trajs)
        # Compute all populations at once using numpy operations
        real_parts = all_trajs_np[:, :, :6, 0]  # Shape: (n_trajs, n_times, 6)
        imag_parts = all_trajs_np[:, :, 6:2*6, 0]  # Shape: (n_trajs, n_times, 6)
        # Compute populations: |psi|^2 = Re^2 + Im^2
        populations = real_parts**2 + imag_parts**2  # Shape: (n_trajs, n_times, 6)
        # Vectorized mean calculation
        mean_populations = np.mean(populations, axis=0)  # Shape: (n_times, 6)
        # Create figure with one subplot
        _, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        # Plot mean population trajectories
        state_labels = ["|G>", "|X_H>", "|X_V>", "|D_H>", "|D_V>", "|B>"]
        for i in range(6):
            ax1.plot(self.t_array, mean_populations[:, i], 
                    label=f"|{i}> = {state_labels[i]}")
        ax1.set_xlabel("Time (ps)")
        ax1.set_ylabel("Population")
        ax1.set_title("Mean Population Trajectories")
        ax1.legend()
        ax1.grid()
        plt.tight_layout()
        plt.show()

    def plot_mean_dressed_population_trajectories(self):
        """Plot mean dressed_state population trajectories for all states."""
        # Convert to numpy array once
        all_trajs_np = np.array(self.trajs)
        # define the dressed-state transformation and convert it to a real representation
        _, _, dressed_trans = create_dressed_energies_and_states(self.params_QD)
        dressed_trans_real = np.array(complex_to_real_block(dressed_trans))    
        all_trajs_dressed = dressed_trans_real @ all_trajs_np
        # Compute all populations at once using numpy operations
        real_parts = all_trajs_dressed[:, :, :6]  # Shape: (n_trajs, n_times, 6)
        imag_parts = all_trajs_dressed[:, :, 6:2*6]  # Shape: (n_trajs, n_times, 6)
        # Compute populations: |psi|^2 = Re^2 + Im^2
        populations = real_parts**2 + imag_parts**2  # Shape: (n_trajs, n_times, 6)
        # Vectorized mean calculation
        mean_populations = np.mean(populations, axis=0)  # Shape: (n_times, 6)
        # Create figure with one subplot
        _, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        # Plot mean population trajectories
        state_labels = ["|G>", "|X_H>", "|X_V>", "|D_H>", "|D_V>", "|B>"]
        for i in range(6):
            ax1.plot(self.t_array, mean_populations[:, i], 
                    label=f"|{i}> = {state_labels[i]}")
        ax1.set_xlabel("Time (ps)")
        ax1.set_ylabel("Dressed-State Population")
        ax1.set_title("Mean Dressed Population Trajectories")
        ax1.legend()
        ax1.grid()
        plt.tight_layout()
        plt.show()
    
    def plot_control_field(self, title):
        if callable(self.control_FF):
            control_FF_array = self.control_FF(self.t_array)
        elif isinstance(self.control_FF, (list, np.ndarray, jnp.ndarray)):
            control_FF_array = self.control_FF
        else:
            raise TypeError("control_FF must be either a callable or an array (list or numpy.ndarray)")
        plt.figure()
        plt.plot(self.t_array, np.asarray(control_FF_array))
        plt.xlabel("Time (ps)")
        plt.ylabel("Control Field (meV)")
        plt.title(title)
        plt.show()
    
    def plot_control_field_fft(self, title):
        if callable(self.control_FF):
            control_FF_array = self.control_FF(self.t_array)
        elif isinstance(self.control_FF, (list, np.ndarray, jnp.ndarray)):
            control_FF_array = self.control_FF
        else:
            raise TypeError("control_FF must be either a callable or an array (list or numpy.ndarray)")
        control_FF_FFT = np.fft.fft(np.asarray(control_FF_array))
        control_FF_FFT = np.abs(control_FF_FFT)
        control_FF_FFT = control_FF_FFT[:len(control_FF_FFT)//2]
        plt.figure()
        plt.plot(control_FF_FFT)
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.show()
