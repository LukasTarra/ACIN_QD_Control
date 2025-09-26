"""import packages"""
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, vmap, block_until_ready
from jax.lax import scan
from jax.scipy.optimize import minimize
from getparameters import get_parameters

"""#might is the keyword for todos"""

"""global variables"""
N_STATES = 6 #number of states
# RNG
SEED = 42
RNG = np.random.default_rng(SEED)
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
        np.sqrt(dark_state_decay_rate) * (ground_state @ dark_exciton_x.conj().T),
        np.sqrt(dark_state_decay_rate) * (ground_state @ dark_exciton_y.conj().T),
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
    M_real = M.real
    M_imag = M.imag
    return np.block([[M_real, -M_imag], [M_imag, M_real]])

def plot_population_trajectories(results):
    plt.figure(figsize=(10, 6))
    state_labels = ["|G>", "|X_H>", "|X_V>", "|D_H>", "|D_V>", "|B>"]
    for i in range(N_STATES):
        plt.plot(results.times, results.expect[i], label=(f"|{i}> = " + state_labels[i]))

    plt.xlabel("Time (ps)")
    plt.ylabel("Population")
    plt.title("Population Trajectories")
    plt.legend()
    plt.grid()
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

def create_jax_noise_traj_arrays(number_trajectories_opt, number_steps, dt):
    # Erzeuge feste Rauschpfade für die Optimierung (reproduzierbar)
    dW_real_opt_val = RNG.normal(size=(number_trajectories_opt, number_steps)) * np.sqrt(dt)
    dW_imag_opt_val = RNG.normal(size=(number_trajectories_opt, number_steps)) * np.sqrt(dt)
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
    H_0 = H_QD + H_bx + H_bz
    # Add control Hamiltonians (factor for polarization overlap included)
    H_c_H = pol_overlaps["H"] * params.hbar * (exciton_x @ ground_state.conj().T + ground_state @ exciton_x.conj().T + exciton_x @ biexciton.conj().T + biexciton @ exciton_x.conj().T)
    H_c_V = pol_overlaps["V"] * params.hbar * (exciton_y @ ground_state.conj().T + ground_state @ exciton_y.conj().T + exciton_y @ biexciton.conj().T + biexciton @ exciton_y.conj().T)
    H_control = H_c_H + H_c_V
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
    LdagL_operators_j = jnp.stack( [jnp.array(LdagL) for LdagL in LdagL_operators_real] )
    # L_operators_j = [jnp.array(L) for L in L_operators_real]
    # LdagL_operators_j = [jnp.array(LdagL) for LdagL in LdagL_operators_real]
    psi_0_j = jnp.array(psi_0_real)
    psi_T_j = jnp.array(psi_T_real)
    I_imag_j = jnp.array(I_imag_real)

    return H_0_j, H_control_j, L_operators_j, LdagL_operators_j, psi_0_j, psi_T_j, I_imag_j

@jit
def normalize_psi(psi):
    regularization = 1e-12 # prevent division by zero
    norm = jnp.linalg.norm(psi)
    return psi / (norm + regularization)

# @jit
# def em_step(psi, dW_real, dW_imag, u, dt, H_0_j, H_control_j, L_operators_j, LdagL_operators_j, I_imag_j):
#     # number of collapse operators
#     N_collapse_operators = len(L_operators_j)
    
#     # Normalize input
#     psi_n = normalize_psi(psi)
#     H_total = H_0_j + u * H_control_j

#     # 1. Deterministic drift part (Drift-Term)
#     def f_drift(p):
#         L_avg_j = [p.conj().T @ (L_operators_j[i] @ p) for i in range(N_collapse_operators)]
#         return -(I_imag_j/HBAR) @ (H_total @ p) - 0.5* sum([ LdagL_operators_j[i]@ p -2* L_avg_j[i]*L_operators_j[i] @ p + L_avg_j[i]**2 * p for i in range(len(LdagL_operators_j)) ])
#     # Midpoint-Methode (RK2) für den Drift
#     k1 = f_drift(psi_n)
#     drift = f_drift(psi_n + 0.5 * dt * k1) # This is the slope at the midpoint

#     # 2. Stochastic diffusion part (Diffusion-Term)
#     L_psi = [ L_operators_j[i] @ psi_n for i in range(N_collapse_operators) ]
#     avg_L = [ psi_n.conj().T @ L_psi[i] for i in range(N_collapse_operators)]
#     diffusion_base = [ L_psi[i] - avg_L[i] * psi_n for i in range(N_collapse_operators) ]
#     # # correct implementation with different dWs # might restore this
#     # diffusion = sum([ diffusion_base[i] * dW_real[i] + (I_imag_j @ diffusion_base[i]) * dW_imag[i] for i in range(len(diffusion_base)) ])
#     #test implementation with same dWs
#     diffusion = sum([ diffusion_base[i] * dW_real + (I_imag_j @ diffusion_base[i]) * dW_imag for i in range(N_collapse_operators) ])

#     # Combine using Euler-Maruyama for the full SDE
#     psi_next = psi_n + drift * dt + diffusion
#     psi_next = normalize_psi(psi_next)
#     return psi_next

@jit
def em_step(psi, dW_real, dW_imag, u, dt, H_0_j, H_control_j, L_operators_j, LdagL_operators_j, I_imag_j):
    # Normalize input
    psi_n = normalize_psi(psi)
    H_total = H_0_j + u * H_control_j

    # 1. Deterministic drift part (vectorized)
    def f_drift(p):
         # Vectorized expectation values: <L_i> = psi_dag @ L_i @ psi
         L_psi = L_operators_j @ p  # Shape: (N_collapse, dim, 1)
         L_avg = jnp.sum(p.conj().T * L_psi, axis=(1,2))  # Shape: (N_collapse,)
        
         # Vectorized drift calculation
         LdagL_psi = LdagL_operators_j @ p  # Shape: (N_collapse, dim, 1)
         term1 = -0.5 * jnp.sum(LdagL_psi, axis=0)  # Sum over collapse operators
         term2 = jnp.sum(L_avg[:, None, None] * L_operators_j @ p, axis=0)
         term3 = -0.5 * jnp.sum(L_avg[:, None, None]**2 * p, axis=0)
        
         return -(I_imag_j/HBAR) @ (H_total @ p) + term1 + term2 + term3
    
    # Midpoint method (RK2) for drift
    k1 = f_drift(psi_n)
    drift = f_drift(psi_n + 0.5 * dt * k1)

    # 2. Stochastic diffusion part (vectorized)
    L_psi = L_operators_j @ psi_n  # Shape: (N_collapse, dim, 1)
    L_avg = jnp.sum(psi_n.conj().T * L_psi, axis=(1,2))  # Shape: (N_collapse,)
    diffusion_base = L_psi - L_avg[:, None, None] * psi_n  # Shape: (N_collapse, dim, 1)
    # # Use the correct implementation with different dWs (more accurate)
    # diffusion = jnp.sum(
    #     diffusion_base * dW_real[:, None, None] + 
    #     (I_imag_j @ diffusion_base) * dW_imag[:, None, None], 
    #     axis=0
    # )
    # Use the approx implementation with same dWs (less accurate)
    diffusion = jnp.sum(
        diffusion_base * dW_real + 
        (I_imag_j @ diffusion_base) * dW_imag, 
        axis=0
    )
    # Combine using Euler-Maruyama
    psi_next = psi_n + drift * dt + diffusion
    return normalize_psi(psi_next)

@jit
def simulate_single_traj(u_traj, dW_real_traj, dW_imag_traj, psi_0_j, dt, H_0_j, H_control_j, L_operators_j, LdagL_operators_j, I_imag_j):
    def step_fn(carry, inputs):
        psi = carry
        dW_r, dW_i, u = inputs
        psi_next = em_step(psi, dW_r, dW_i, u, dt, H_0_j, H_control_j, L_operators_j, LdagL_operators_j, I_imag_j)
        return psi_next, psi_next

    inputs = (dW_real_traj, dW_imag_traj, u_traj)
    final_state, traj = scan(step_fn, psi_0_j, inputs)
    
    return jnp.concatenate([psi_0_j[None], traj])

# vmap over trajectories
sim_forward_vmap = vmap(simulate_single_traj, in_axes=(None, 0, 0, None, None, None, None, None, None, None), out_axes=0)

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

if __name__ == "__main__":
    # Load parameters
    par_QD = get_parameters()
    # Polarization overlaps (e_H * e_L, e_V * e_L)
    polarization_overlaps = {"H": 1, "V": 0}
    # Choose initial state (G, X_H, X_V, D_H, D_V, B)
    init_state = "B"
    target_state = "D_H"
    # no. of Trajectories for optimization & simulation
    number_trajectories_opt = 100
    number_trajectories_sim = 100
    # set control weight
    control_weight = 0.0001

    # Define time array for the simulation
    t_start = 0
    t_end = 1000  # ps
    dt = 0.05      # time step in ps
    t_array = np.linspace(t_start, t_end, int((t_end - t_start) / dt) + 1)

    # Define control input guess
    control_FF_guess = lambda t: 1000 * (1/(1+ t**2/100)) * np.sin(2 * np.pi * t)
    control_FF_guess_array = control_FF_guess(t_array)
    # control_FF_guess_array = np.zeros(len(t_array))
    plot_control_field(control_FF_guess, t_array)
    plot_control_field_fft( control_FF_guess, t_array )

    # set up JAX simulation
    H_0_j, H_control_j, L_operators_j, LdagL_operators_j, psi_0_j, psi_T_j, I_imag_j = jax_sim_setup(init_state,target_state,polarization_overlaps,par_QD)

    # # optimize control guess
    # dW_real_opt_j, dW_imag_opt_j = create_jax_noise_traj_arrays(number_trajectories_opt, len(t_array), dt)
    # control_FF_opt_bfgs = optimize_u_traj(control_FF_guess_array, control_weight, dW_real_opt_j, dW_imag_opt_j, psi_0_j, psi_T_j, dt, H_0_j, H_control_j, L_operators_j, LdagL_operators_j, I_imag_j)

    # simulate the system with the guess
    dW_real_opt_j, dW_imag_opt_j = create_jax_noise_traj_arrays(number_trajectories_sim, len(t_array), dt)
    all_trajs = sim_forward_vmap(control_FF_guess_array, dW_real_opt_j, dW_imag_opt_j, psi_0_j, dt, H_0_j, H_control_j, L_operators_j, LdagL_operators_j, I_imag_j)
    traj_mean = jnp.mean(all_trajs[:,:,:,0],axis=0)
    traj_mean = np.array(traj_mean)
    print(traj_mean[:10,:])
    # plt.plot(sum([ traj_mean[:, i]**2 for i in range(traj_mean.shape[1]) ]))
    plt.plot(traj_mean[:, 5]**2 + traj_mean[:, 11]**2  )
    plt.show()

