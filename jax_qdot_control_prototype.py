"""import packages"""
import time
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, vmap, profiler
import jax.debug as jdebug
from jax.lax import scan
from jax.scipy.optimize import minimize
from jax.scipy.linalg import expm
from functools import partial
from getparameters import get_parameters

# broader shell output before linebreaks for debugging 
np.set_printoptions(linewidth=300, edgeitems=10)

"""#might is the keyword for todos"""

"""global variables"""
N_STATES = 6 #number of states
# RNG
SEED = 42
# RNG = np.random.default_rng(SEED)
RNG = np.random
nominal_parameters = get_parameters()
HBAR = nominal_parameters.hbar # Planck constant in weird units


"""START helper functions"""

def create_QD_hamiltonian_terms_and_states(params):
    E_X_H = params.E_X_H  # Exciton horizontal energy
    E_X_V = params.E_X_V  # Exciton vertical energy
    E_D_H = params.E_D_H  # Dark exciton horizontal energy
    E_D_V = params.E_D_V  # Dark exciton vertical energy
    E_B = params.E_XX     # Biexciton energy
    mu_b = params.mu_B    # Bohr magneton
    bx = params.B_x       # Magnetic field in x direction
    bz = params.B_z       # Magnetic field in z direction
    g_ex = params.g_ex    # Electron g factor
    g_hx = params.g_hx    # Hole g factor
    g_ez = 0              # Placeholder for electron g factor in z direction
    g_hz = 0              # Placeholder for hole g factor in z direction

    # Define the six-level system
    if N_STATES != 6:
        raise ValueError("N_states must be 6 for a six-level system")
    ground_state = np.zeros((N_STATES,1), dtype=np.complex128)
    ground_state[0] = 1
    exciton_x = np.zeros((N_STATES,1), dtype=np.complex128)
    exciton_x[1] = 1
    exciton_y = np.zeros((N_STATES,1), dtype=np.complex128)
    exciton_y[2] = 1
    dark_exciton_x = np.zeros((N_STATES,1), dtype=np.complex128)
    dark_exciton_x[3] = 1
    dark_exciton_y = np.zeros((N_STATES,1), dtype=np.complex128)
    dark_exciton_y[4] = 1
    biexciton = np.zeros((N_STATES,1), dtype=np.complex128)
    biexciton[5] = 1

    #np.zeros((N_STATES,1))

    # Define the Hamiltonian terms
    H_QD = E_X_H * exciton_x @ exciton_x.conj().T + E_X_V * exciton_y @ exciton_y.conj().T + \
         E_D_H * dark_exciton_x @ dark_exciton_x.conj().T + E_D_V * dark_exciton_y @ dark_exciton_y.conj().T + \
         E_B * biexciton @ biexciton.conj().T

    # Bright-dark coupling depending on Bx
    if bx != 0:
        H_bx = -0.5 * mu_b * bx * (g_hx + g_ex) * (exciton_x @ dark_exciton_x.conj().T + dark_exciton_x @ exciton_x.conj().T) + \
              -0.5 * mu_b * bx * (g_hx - g_ex) * (exciton_y @ dark_exciton_y.conj().T + dark_exciton_y @ exciton_y.conj().T)
    else:
        H_bx = np.zeros((N_STATES, N_STATES))

    # Bright-bright and dark-dark coupling depending on Bz
    if bz != 0.0:
        H_bz = 1j * 0.5 * mu_b * bz * (g_ez - 3 * g_hz) * (exciton_x @ exciton_y.conj().T - exciton_y @ exciton_x.conj().T) + \
              1j * -0.5 * mu_b * bz * (g_ez + 3 * g_hz) * (dark_exciton_x @ dark_exciton_y.conj().T - dark_exciton_y @ dark_exciton_x.conj().T)
    else:
        H_bz = np.zeros((N_STATES, N_STATES))

    return H_QD, H_bx, H_bz, ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton


def create_control_field(times, control_input):
    """
    Create a control field function from various input types.

    Args:
        times: Array of time points
        control_input: Either a callable function, a numpy array of control values,
                      or an unsupported type (which defaults to zero)

    Returns:
        A function that takes a time t and returns the control field value at that time
    """
    zero_function = lambda t: 0.0

    if callable(control_input):
        # If control_input is already a function, use it directly
        return control_input

    elif isinstance(control_input, np.ndarray):
        if len(control_input) == len(times):
            # For exact length match, create a dictionary for O(1) lookup
            # This is more efficient than np.where() which is O(n)
            time_value_map = {t: val for t, val in zip(times, control_input)}
            return lambda t: time_value_map.get(t, 0.0)
        else:
            # For different lengths, use interpolation
            # np.interp is efficient with O(log n) complexity
            return lambda t: np.interp(t, times, control_input)

    else:
        # For unsupported types, return a zero function
        print("Warning: Control field's type not supported, set it to 0.")
        return zero_function

def create_collapse_operators(ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton, params):
    # Define decay rates for each state
    exciton_decay_rate = 1 / params.Gamma_X_inv  # Exciton decay rate
    dark_state_decay_rate = 0  # Decay rate for dark states
    biexciton_decay_rate = 1 / params.Gamma_XX_inv  # Biexciton decay rate

    # Create collapse operators with their respective decay rates
    collapse_ops = [
        np.sqrt(exciton_decay_rate) * (ground_state @ exciton_x.conj().T),
        np.sqrt(exciton_decay_rate) * (ground_state @ exciton_y.conj().T),
        # # Leave out the dark states to reduce overhead
        # np.sqrt(dark_state_decay_rate) * (ground_state @ dark_exciton_x.conj().T),
        # np.sqrt(dark_state_decay_rate) * (ground_state @ dark_exciton_y.conj().T),
        np.sqrt(biexciton_decay_rate) * (exciton_x @ biexciton.conj().T),
        np.sqrt(biexciton_decay_rate) * (exciton_y @ biexciton.conj().T)
    ]

    return collapse_ops

def create_initial_state(psi_0_choice, ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton):
    match psi_0_choice:
        case "G":
            psi_0 = ground_state
        case "X_H":
            psi_0 = exciton_x
        case "X_V":
            psi_0 = exciton_y
        case "D_H":
            psi_0 = dark_exciton_x
        case "D_V":
            psi_0 = dark_exciton_y
        case "B":
            psi_0 = biexciton

    print("The chosen initial state is: ", psi_0_choice)

    return psi_0

def create_target_state(psi_T_choice, ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton):
    match psi_T_choice:
        case "G":
            psi_T = ground_state
        case "X_H":
            psi_T = exciton_x
        case "X_V":
            psi_T = exciton_y
        case "D_H":
            psi_T = dark_exciton_x
        case "D_V":
            psi_T = dark_exciton_y
        case "B":
            psi_T = biexciton

    print("The chosen target state is: ", psi_T_choice)

    return psi_T

# komplexe Matrizen als reelle Blockmatrix
def complex_to_real_block(M: np.ndarray) -> np.ndarray:
    """Convert a complex matrix to a real block matrix representation.

    This function transforms a complex matrix M into a real matrix by
    representing complex numbers as 2x2 real blocks:
    [real(M)  -imag(M)]
    [imag(M)   real(M)]

    The resulting matrix has dtype float64 to ensure all elements are real.
    """
    # Extract real and imaginary parts
    M_real = M.real
    M_imag = M.imag
    # Create the real block matrix
    # Using np.block for clear block structure
    block_matrix = jnp.block([
        [M_real, -M_imag],
        [M_imag, M_real]
    ])
    # Explicitly cast to real dtype (float32) to remove any imaginary components
    return block_matrix.astype(np.float32)

def real_to_complex_block(M: np.ndarray) -> np.ndarray:
    """Convert a real block matrix back to a complex matrix.
    
    This function transforms a real block matrix M back to its complex form
    by extracting the real and imaginary parts from the block structure:
    M_complex = M_upper_left + 1j * M_lower_left
    """
    # Get the dimensions of the input matrix
    n, m = M.shape
    # The real block matrix has twice the dimensions of the original complex matrix
    # So the original complex matrix has dimensions n/2 × m/2
    n_half = n // 2
    m_half = m // 2
    # Extract the real part (upper left block)
    real_part = M[:n_half, :m_half]
    # Extract the imaginary part (bottom left block)
    imag_part = M[n_half:, :m_half]
    # Combine them to form the complex matrix
    complex_matrix = real_part + 1j * imag_part
    return complex_matrix

def plot_population_trajectories(all_trajs,t_array):
    """Plot population trajectories for all states (mean) and multiple (randomly chosen) trajectories."""
    # Convert to numpy array once
    all_trajs_np = np.array(all_trajs)
    # Compute all populations at once using numpy operations
    real_parts = all_trajs_np[:, :, :N_STATES, 0]  # Shape: (n_trajs, n_times, N_STATES)
    imag_parts = all_trajs_np[:, :, N_STATES:2*N_STATES, 0]  # Shape: (n_trajs, n_times, N_STATES)
    # Compute populations: |ψ|² = Re² + Im²
    populations = real_parts**2 + imag_parts**2  # Shape: (n_trajs, n_times, N_STATES)
    # Vectorized mean calculation
    mean_populations = np.mean(populations, axis=0)  # Shape: (n_times, N_STATES)
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    # Plot mean population trajectories
    state_labels = ["|G>", "|X_H>", "|X_V>", "|D_H>", "|D_V>", "|B>"]
    for i in range(N_STATES):
        ax1.plot(t_array, mean_populations[:, i], 
                label=f"|{i}> = {state_labels[i]}")
    ax1.set_xlabel("Time (ps)")
    ax1.set_ylabel("Population")
    ax1.set_title("Mean Population Trajectories")
    ax1.legend()
    ax1.grid()

    # Randomly select 20 trajectories to plot
    n_selected = min(10, all_trajs_np.shape[0])  # Handle case with fewer than 10 trajectories
    selected_indices = np.random.choice(all_trajs_np.shape[0], size=n_selected, replace=False)
    # Plot individual trajectories more efficiently
    selected_populations = populations[selected_indices, :, :]  # Shape: (n_selected, n_times, N_STATES)

    for i in range(N_STATES):
        # Plota all selected trajectories for this state at once
        for j in range(n_selected):
            ax2.plot(t_array, selected_populations[j, :, i],
                 alpha=0.5, color=plt.cm.tab20(i * 2))
        # Add label for the state
        ax2.plot([], [], color=plt.cm.tab20(i * 2), label=state_labels[i], alpha=0.5)
    ax2.set_xlabel("Time (ps)")
    ax2.set_ylabel("Population")
    ax2.set_title("Individual Population Trajectories")
    ax2.legend()
    ax2.grid()
    plt.tight_layout()
    plt.show()

def plot_control_field(control_FF, t_array):
    if callable(control_FF):
        control_FF_array = control_FF(t_array)
    elif isinstance(control_FF, np.ndarray):
        control_FF_array = control_FF
    else:
        raise ValueError("Control field must be either a callable function or a numpy array")
    plt.figure()
    plt.plot(t_array, control_FF_array)
    plt.xlabel("Time (ps)")
    plt.ylabel("Control Field (meV)")
    plt.title("Control Field")
    plt.show()

def plot_control_field_fft(control_FF, t_array):
    if callable(control_FF):
        control_FF_array = control_FF(t_array)
    elif isinstance(control_FF, np.ndarray):
        control_FF_array = control_FF
    else:
        raise ValueError("Control field must be either a callable function or a numpy array")
    control_FF_FFT = np.fft.fft(control_FF_array)
    control_FF_FFT = np.abs(control_FF_FFT)
    control_FF_FFT = control_FF_FFT[:len(control_FF_FFT)//2]
    plt.figure()
    plt.plot(control_FF_FFT)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Control Field FFT")
    plt.show()

def create_jax_noise_traj_arrays(number_collapse, number_trajectories, number_steps, dt):
    # Erzeuge feste Rauschpfade für die Optimierung (reproduzierbar)
    dW_real_opt_val = RNG.normal(size=(number_trajectories, number_steps, number_collapse)) * np.sqrt(dt)
    dW_imag_opt_val = RNG.normal(size=(number_trajectories, number_steps, number_collapse)) * np.sqrt(dt)
    # Konvertiere zu JAX
    dW_real_opt_j = jnp.array(dW_real_opt_val)
    dW_imag_opt_j = jnp.array(dW_imag_opt_val)

    return dW_real_opt_j, dW_imag_opt_j

"""END helper functions"""


"""START simulation functions"""

def jax_sim_setup(psi_0_choice,psi_T_choice,pol_overlaps,params):

    # Create Hamiltonian terms
    H_QD, H_bx, H_bz, ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton = create_QD_hamiltonian_terms_and_states(params)
    # Total Hamiltonian (without control fields)
    H_0 = (H_QD + H_bx + H_bz) / params.hbar / 10
    # Add control Hamiltonians (factor for polarization overlap included)
    H_c_H = pol_overlaps["H"] * params.hbar * (exciton_x @ ground_state.conj().T + ground_state @ exciton_x.conj().T + exciton_x @ biexciton.conj().T + biexciton @ exciton_x.conj().T)
    H_c_V = pol_overlaps["V"] * params.hbar * (exciton_y @ ground_state.conj().T + ground_state @ exciton_y.conj().T + exciton_y @ biexciton.conj().T + biexciton @ exciton_y.conj().T)
    H_control = (H_c_H + H_c_V) / params.hbar
    # Create collapse operators
    L_operators = create_collapse_operators(ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton, params)
    LdagL_operators = [L.conj().T @ L for L in L_operators]
    # Create initial state
    psi_0 = create_initial_state(psi_0_choice, ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton)
    # Create target state
    psi_T = create_target_state(psi_T_choice, ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton)
    # convert to real blocks / vectors
    H_0_real = complex_to_real_block(H_0)
    H_control_real = complex_to_real_block(H_control)
    L_operators_real = [complex_to_real_block(L) for L in L_operators]
    LdagL_operators_real = [complex_to_real_block(LdagL) for LdagL in LdagL_operators]
    psi_0_real = np.concatenate([psi_0.real, psi_0.imag])
    psi_T_real = np.concatenate([psi_T.real, psi_T.imag])
    I_imag_real = np.block([[np.zeros((N_STATES, N_STATES)), -np.eye(N_STATES)], [np.eye(N_STATES), np.zeros((N_STATES, N_STATES))]])
    # convert real arrays to JAX arrays (L operators are stacked)
    H_0_j = jnp.array(H_0_real)
    H_control_j = jnp.array(H_control_real)
    L_operators_j = jnp.stack( [jnp.array(L) for L in L_operators_real] )
    L_operators_transposed_j = L_operators_j.transpose((0,2,1))
    LdagL_operators_j = jnp.stack( [jnp.array(LdagL) for LdagL in LdagL_operators_real] )
    # L_operators_j = [jnp.array(L) for L in L_operators_real]
    # LdagL_operators_j = [jnp.array(LdagL) for LdagL in LdagL_operators_real]
    psi_0_j = jnp.array(psi_0_real)
    psi_T_j = jnp.array(psi_T_real)
    I_imag_j = jnp.array(I_imag_real)

    return H_0_j, H_control_j, L_operators_j, L_operators_transposed_j, LdagL_operators_j, psi_0_j, psi_T_j, I_imag_j

def transform_jax_to_rotating_frame(t_array, H_0_j, H_control_j, L_operators_j, L_operators_transposed_j, LdagL_operators_j):

    # Compute H_0_complex_times_t with dimensions (len(t_array), n, n)
    H_0_complex         = real_to_complex_block(H_0_j)
    H_0_complex_times_t = jnp.einsum('ij,t->tij', H_0_complex, t_array)
    # Compute unitary transformation U (matrix exponential)
    U                   = expm(-1j * H_0_complex_times_t)
    U_dag               = U.conj().transpose((0,2,1)) # Adjoint of unitary transform
    # transform the relevant matrices to rotating frame
    H_control_complex   = real_to_complex_block(H_control_j)
    H_control_tilde     = U_dag @ H_control_complex @ U # Rotating frame Hamiltonian
    dim = H_control_j.shape[-1] # dimension 
    dim_half = H_control_j.shape[-1] // 2
    N_collapse = L_operators_j.shape[0]
    L_operators_tilde = np.zeros((len(t_array),N_collapse,dim_half,dim_half),dtype=np.complex128)
    L_operators_dag_tilde = np.zeros((len(t_array),N_collapse,dim_half,dim_half),dtype=np.complex128)
    LdagL_operators_tilde = np.zeros((len(t_array),N_collapse,dim_half,dim_half),dtype=np.complex128)
    for k in range(N_collapse):
        L_operator_complex = real_to_complex_block(L_operators_j[k,:,:])
        L_operator_dag_complex = real_to_complex_block(L_operators_transposed_j[k,:,:])
        LdagL_operator_complex = real_to_complex_block(LdagL_operators_j[k,:,:])
        # Apply unitary transformation to L operators
        L_operators_tilde[:,k,:,:] = U_dag @ L_operator_complex @ U
        L_operators_dag_tilde[:,k,:,:] = U_dag @ L_operator_dag_complex @ U
        LdagL_operators_tilde[:,k,:,:] = U_dag @ LdagL_operator_complex @ U
    # transform back to the real representations using a vmap for time
    time_matrices_complex_to_real_block_vmap = vmap(complex_to_real_block, in_axes=(0), out_axes=0)
    H_control_tilde_j = time_matrices_complex_to_real_block_vmap(H_control_tilde)
    L_operators_tilde_j = np.zeros((len(t_array),N_collapse,dim,dim))
    L_operators_transposed_tilde_j = np.zeros((len(t_array),N_collapse,dim,dim))
    LdagL_operators_tilde_j = np.zeros((len(t_array),N_collapse,dim,dim))
    for k in range(N_collapse):
        L_operators_tilde_j[:,k,:,:] = time_matrices_complex_to_real_block_vmap(L_operators_tilde[:,k,:,:])
        L_operators_transposed_tilde_j[:,k,:,:] = time_matrices_complex_to_real_block_vmap(L_operators_dag_tilde[:,k,:,:])
        LdagL_operators_tilde_j[:,k,:,:] = time_matrices_complex_to_real_block_vmap(LdagL_operators_tilde[:,k,:,:])
    # convert resulting arrays to JAX arrays
    U_j = jnp.array( time_matrices_complex_to_real_block_vmap(U) )
    U_dag_j = jnp.array( time_matrices_complex_to_real_block_vmap(U_dag) )
    H_control_tilde_j = jnp.array(H_control_tilde_j)
    L_operators_tilde_j = jnp.array(L_operators_tilde_j)
    L_operators_transposed_tilde_j = jnp.array(L_operators_transposed_tilde_j)
    LdagL_operators_tilde_j = jnp.array(LdagL_operators_tilde_j)

    return U_j, U_dag_j, H_control_tilde_j, L_operators_tilde_j, L_operators_transposed_tilde_j, LdagL_operators_tilde_j

@jit
def normalize_psi(psi):
    regularization = 1e-12 # prevent division by zero
    norm = jnp.linalg.norm(psi)
    return psi / (norm + regularization)

# @partial(jit, static_argnames=['dt', 'H_0_j', 'I_imag_j'])
def em_step(psi_n, dW_real, dW_imag, u, dt, H_control_j, L_operators_j, L_operators_transposed_j, LdagL_operators_j, I_imag_j):
    """
    Optimized Euler-Maruyama step in the rotating frame.
    """
    # Pre-compute total Hamiltonian
    H_total = u * H_control_j

    def f_drift(p):
        # More efficient computation without temporary arrays
        # Compute L_unitary = 0.5*(L + L_dag) once
        L_unitary = 0.5 * (L_operators_j + L_operators_transposed_j)
        # Compute L_unitary @ psi for all operators at once
        L_unitary_psi = L_unitary @ p  # Shape: (N_collapse, dim, 1)
        # Compute expectation values <L_unitary> = psi^dagger @ L_unitary @ psi
        # Reshape psi to (1, dim, 1) for proper broadcasting
        psi_reshaped = p.reshape(1, -1, 1)  # Shape: (1, dim, 1)
        L_unitary_avg = jnp.sum(psi_reshaped * L_unitary_psi, axis=(1, 2), keepdims=True)  # Shape: (N_collapse, 1, 1)
        # Compute LdagL @ psi for all operators at once
        LdagL_psi = LdagL_operators_j @ p  # Shape: (N_collapse, dim, 1)
        # Compute all terms efficiently
        term1 = -0.5 * jnp.sum(LdagL_psi, axis=0)  # Sum over collapse operators
        # Compute L @ psi for all operators at once
        L_psi = L_operators_j @ p  # Shape: (N_collapse, dim, 1)
        # term2 = sum(<L_unitary> * L @ psi)
        term2 = jnp.sum(L_unitary_avg * L_psi, axis=0)
        # term3 = -0.5 * sum(<L_unitary>^2 * psi)
        term3 = -0.5 * jnp.sum(L_unitary_avg**2 * p, axis=0)
        # Hamiltonian term
        hamiltonian_term = -I_imag_j @ (H_total @ p)

        return hamiltonian_term + term1 + term2 + term3

    # 1. Deterministic drift part (Runge-Kutta 2nd order)
    k1 = f_drift(psi_n)
    drift = f_drift(psi_n + 0.5 * dt * k1)
    # Compute intermediate state
    psi_interm = normalize_psi(psi_n + drift * dt)

    # 2. Stochastic diffusion part (optimized)
    L_unitary = 0.5 * (L_operators_j + L_operators_transposed_j)
    L_unitary_psi_interm = L_unitary @ psi_interm
    # Compute expectation values for intermediate state
    psi_interm_reshaped = psi_interm.reshape(1, -1, 1)
    L_unitary_avg_interm = jnp.sum(psi_interm_reshaped * L_unitary_psi_interm, 
                                  axis=(1, 2), keepdims=True)
    # Compute L @ psi_interm
    L_psi_interm = L_operators_j @ psi_interm
    # Compute diffusion base: L@psi - <L> * psi
    diffusion_base = L_psi_interm - L_unitary_avg_interm * psi_interm
    # Apply stochastic terms with proper broadcasting
    # dW_real and dW_imag have shape (N_collapse,), we need to broadcast to match diffusion_base
    dW_real_broadcast = dW_real.reshape(-1, 1, 1)  # Shape: (N_collapse, 1, 1)
    dW_imag_broadcast = dW_imag.reshape(-1, 1, 1)  # Shape: (N_collapse, 1, 1)
    # Compute diffusion term efficiently
    real_part = jnp.sum(diffusion_base * dW_real_broadcast, axis=0)
    imag_part = jnp.sum((I_imag_j @ diffusion_base) * dW_imag_broadcast, axis=0)
    diffusion = real_part + imag_part
    # Combine using Euler-Maruyama
    psi_next = psi_interm + diffusion
    return normalize_psi(psi_next)

# def simulate_single_traj(u_traj, dW_real_traj, dW_imag_traj, psi_0_j, dt, H_0_j, H_control_j, L_operators_j, L_operators_transposed_j, LdagL_operators_j, I_imag_j):
#     def step_fn(carry, inputs):
#         psi = carry
#         dW_r, dW_i, u = inputs
#         psi_next = em_step(psi, dW_r, dW_i, u, dt, H_0_j, H_control_j, L_operators_j, L_operators_transposed_j, LdagL_operators_j, I_imag_j)
#         return psi_next, psi_next

#     inputs = (dW_real_traj, dW_imag_traj, u_traj)
#     final_state, traj = scan(step_fn, psi_0_j, inputs)
    
#     return traj

# # vmap over trajectories
# sim_forward_vmap = (vmap(simulate_single_traj, in_axes=(None, 0, 0, None, None, None, None, None, None, None, None), out_axes=0))

# Simulation in the rotating frame
# @partial(jit, static_argnames=['psi_0_j', 'dt', 'H_control_tilde_j', 'L_operators_tilde_j', 'L_operators_transposed_tilde_j', 'LdagL_operators_tilde_j', 'I_imag_j'])
def simulate_single_traj_rotating_frame(
    u_traj, dW_real_traj, dW_imag_traj, psi_0_j, dt, H_control_tilde_j, L_operators_tilde_j, L_operators_transposed_tilde_j, LdagL_operators_tilde_j, I_imag_j):

    def step_fn(carry, inputs):
        psi = carry
        dW_r, dW_i, u, H_c_tilde_j, L_tilde_j, L_transposed_tilde_j, LdagL_tilde_j = inputs
        psi_next = em_step(psi, dW_r, dW_i, u, dt, H_c_tilde_j, L_tilde_j, L_transposed_tilde_j, LdagL_tilde_j, I_imag_j)
        return psi_next, psi_next

    inputs = (dW_real_traj, dW_imag_traj, u_traj, H_control_tilde_j, L_operators_tilde_j, L_operators_transposed_tilde_j, LdagL_operators_tilde_j)
    _, traj = scan(step_fn, psi_0_j, inputs)
    
    return traj

# Simulation in the rotating frame (optimized)

# @jit
# def simulate_single_traj_rotating_frame(
#     u_traj, dW_real_traj, dW_imag_traj, psi_0_j, dt, 
#     H_control_tilde_j, L_operators_tilde_j, L_operators_transposed_tilde_j, LdagL_operators_tilde_j, I_imag_j):
#     """
#     Optimized version that avoids passing large time-dependent operators through scan.
#     Instead, we use the time index to access operators from precomputed arrays.
#     """
#     H_0_j_rotating = jnp.zeros((len(psi_0_j),len(psi_0_j)))  # Constant zero Hamiltonian in rotating frame
    
#     def step_fn(carry, inputs):
#         psi, time_idx = carry
#         dW_r, dW_i, u = inputs
        
#         # Get time-dependent operators for current time step
#         H_c_tilde_j = H_control_tilde_j[time_idx]
#         L_tilde_j = L_operators_tilde_j[time_idx]
#         L_transposed_tilde_j = L_operators_transposed_tilde_j[time_idx]
#         LdagL_tilde_j = LdagL_operators_tilde_j[time_idx]
        
#         psi_next = em_step(psi, dW_r, dW_i, u, dt, H_0_j_rotating, 
#                           H_c_tilde_j, L_tilde_j, L_transposed_tilde_j, LdagL_tilde_j, I_imag_j)
#         return (psi_next, time_idx + 1), psi_next

#     inputs = (dW_real_traj, dW_imag_traj, u_traj)
#     _, traj = scan(step_fn, (psi_0_j, 0), inputs)
    
#     return traj

# vmap over trajectories
sim_forward_vmap_rotating_frame = (vmap(simulate_single_traj_rotating_frame, in_axes=(None, 0, 0, None, None, None, None, None, None, None), out_axes=0))

def simulate_batch_rotating_frame(u_traj, psi_0_j, dt, U_j, H_control_tilde_j, L_operators_tilde_j, L_operators_transposed_tilde_j, LdagL_operators_tilde_j, I_imag_j, number_collapse_ops, number_trajectories, number_time_steps):

    dW_real_j, dW_imag_j = create_jax_noise_traj_arrays(number_collapse_ops, number_trajectories, number_time_steps, dt)

    all_trajs_rotating = sim_forward_vmap_rotating_frame(
        u_traj, dW_real_j, dW_imag_j, psi_0_j, dt, H_control_tilde_j, L_operators_tilde_j, L_operators_transposed_tilde_j, LdagL_operators_tilde_j, I_imag_j)

    all_trajs = U_j @ all_trajs_rotating   
    return all_trajs

    
"""END simulation functions"""


"""START optimization functions"""

@jit # might remove this
def cost_function(u_traj, control_weight, dW_real_opt_j, dW_imag_opt_j, psi_0_j, psi_T_j, dt, H_0_j, H_control_j, L_operators_j, LdagL_operators_j, I_imag_j):
    all_trajs = sim_forward_vmap(u_traj, dW_real_opt_j, dW_imag_opt_j, psi_0_j, dt, H_0_j, H_control_j, L_operators_j, LdagL_operators_j, I_imag_j) # all_trajs shape: (number_trajectories_opt, steps+1, 2N)
    #final state of every trajectory
    psi_ends = all_trajs[:,-1, :, 0]
    # compute state cost
    state_cost = jnp.abs( jnp.mean(psi_ends, axis=0) - psi_T_j.T )
    expected_state_cost = jnp.mean(state_cost)
    # compute control cost
    control_cost = control_weight * jnp.sum(u_traj**2)

    return expected_state_cost + control_cost

def optimize_u_traj(u_init, control_weight, dW_real_opt_j, dW_imag_opt_j, psi_0_j, psi_T_j, dt, H_0_j, H_control_j, L_operators_j, LdagL_operators_j, I_imag_j):
    print('Start optimization with JAX (BFGS) ...')
    total_cost = lambda u_traj: cost_function(u_traj, control_weight, dW_real_opt_j, dW_imag_opt_j, psi_0_j, psi_T_j, dt, H_0_j, H_control_j, L_operators_j, LdagL_operators_j, I_imag_j)
    res = minimize(total_cost, u_init, method='bfgs')
    u_opt = res.x
    print('\nOptimierung beendet. Erfolg:', res.success, 'Status:', res.status, 'Iterations:', res.nit)
    print('Finale Kosten:', res.fun)

    return u_opt

"""END optimization functions"""


"""START main function"""

def main():
    # Load parameters
    par_QD = get_parameters()
    # Polarization overlaps (e_H * e_L, e_V * e_L)
    polarization_overlaps = {"H": 1, "V": 0}
    # Choose initial state (G, X_H, X_V, D_H, D_V, B)
    init_state = "B"
    target_state = "D_H"
    # no. of Trajectories for optimization & simulation
    number_trajectories_opt = 30
    number_trajectories_sim = 30
    # set control weight
    control_weight = 0.0001

    # Define time array for the simulation
    t_start = 0
    t_end = 1000  # ps
    dt = 0.005      # time step in ps
    t_array = np.linspace(t_start, t_end, int((t_end - t_start) / dt) + 1)

    # Define control input guess
    control_FF_guess = lambda t: 1 * (1/(1+ t**2/100)) * np.sin(2 * np.pi * t)
    # control_FF_guess = lambda t: 0*t
    control_FF_guess_array = control_FF_guess(t_array)
    # control_FF_guess_array = np.zeros(len(t_array))
    # plot_control_field(control_FF_guess, t_array)
    # plot_control_field_fft( control_FF_guess, t_array )

    # set up JAX simulation matrices
    H_0_j, H_control_j, L_operators_j, L_operators_transposed_j, LdagL_operators_j, psi_0_j, psi_T_j, I_imag_j = \
        jax_sim_setup(init_state,target_state,polarization_overlaps,par_QD)
    number_collapse_ops = L_operators_j.shape[0]
    
    # transform to rotating frame
    U_j, U_dag_j, H_control_tilde_j, L_operators_tilde_j, L_operators_transposed_tilde_j, LdagL_operators_tilde_j = \
        transform_jax_to_rotating_frame(t_array, H_0_j, H_control_j, L_operators_j, L_operators_transposed_j, LdagL_operators_j)

    # # optimize control guess
    # dW_real_opt_j, dW_imag_opt_j = create_jax_noise_traj_arrays(number_trajectories_opt, len(t_array), dt)
    # control_FF_opt_bfgs = optimize_u_traj(control_FF_guess_array, control_weight, dW_real_opt_j, dW_imag_opt_j, psi_0_j, psi_T_j, dt, H_0_j, H_control_j, L_operators_j, LdagL_operators_j, I_imag_j)

    # simulate the system with the guess
    start_time = time.time()
    all_trajs = simulate_batch_rotating_frame(control_FF_guess_array, psi_0_j, dt, U_j, H_control_tilde_j, L_operators_tilde_j, L_operators_transposed_tilde_j, LdagL_operators_tilde_j, I_imag_j, number_collapse_ops, number_trajectories_sim, len(t_array))
    all_trajs = np.array(all_trajs)    
    end_time = time.time()
    simulation_time = end_time - start_time
    print(f"Simulation completed in {simulation_time:.2f} seconds")

    # plot population trajectories
    plot_population_trajectories(all_trajs,t_array)

if __name__ == "__main__":
    main()
    # cProfile.run('main()')
