# import packages
import numpy as np
from qutip import basis, Qobj, mesolve
import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d

# user-defined functions & packages
from getparameters import get_parameters

def simulate_dark_states(times,control_input,rho_0_choice,pol_overlaps,params):
    # Load parameters from the params struct
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
    gamma_e = 1/params.Gamma_X_inv  # Exciton decay rate
    gamma_d = 0           # Decay rate for dark states
    gamma_b = 1/params.Gamma_XX_inv # Biexciton decay rate
    hbar = params.hbar    # Planck constant

    # Define the six-level system

    N = 6  # Number of states
    ground_state = basis(N, 0)
    exciton_x = basis(N, 1)
    exciton_y = basis(N, 2)
    dark_exciton_x = basis(N, 3)
    dark_exciton_y = basis(N, 4)
    biexciton = basis(N, 5)

    # Define the Hamiltonian terms
    H_QD = E_X_H * exciton_x * exciton_x.dag() + E_X_V * exciton_y * exciton_y.dag() + \
         E_D_H * dark_exciton_x * dark_exciton_x.dag() + E_D_V * dark_exciton_y * dark_exciton_y.dag() + \
         E_B * biexciton * biexciton.dag()

    # Bright-dark coupling depending on Bx
    if bx != 0:
        H_bx = -0.5 * mu_b * bx * (g_hx + g_ex) * (exciton_x * dark_exciton_x.dag() + dark_exciton_x * exciton_x.dag()) + \
              -0.5 * mu_b * bx * (g_hx - g_ex) * (exciton_y * dark_exciton_y.dag() + dark_exciton_y * exciton_y.dag())
    else:
        H_bx = Qobj(np.zeros((N, N)))

    # Bright-bright and dark-dark coupling depending on Bz
    if bz != 0.0:
        H_bz = 1j * 0.5 * mu_b * bz * (g_ez - 3 * g_hz) * (exciton_x * exciton_y.dag() - exciton_y * exciton_x.dag()) + \
              1j * -0.5 * mu_b * bz * (g_ez + 3 * g_hz) * (dark_exciton_x * dark_exciton_y.dag() - dark_exciton_y * dark_exciton_x.dag())
    else:
        H_bz = Qobj(np.zeros((N, N)))

    # Total Hamiltonian (without control fields)
    H_0 = H_QD + H_bx + H_bz

    # add control Hamiltonians (factor for polarization overlap included)
    H_c_H = pol_overlaps["H"]* hbar* (exciton_x*ground_state.dag()+ground_state*exciton_x.dag() + exciton_x*biexciton.dag()+biexciton*exciton_x.dag()) 
    H_c_V = pol_overlaps["V"]* hbar* (exciton_y*ground_state.dag()+ground_state*exciton_y.dag() + exciton_y*biexciton.dag()+biexciton*exciton_y.dag()) 
    H_c = H_c_H + H_c_V

    #add the control input as a function or numpy array
    if callable(control_input):
        control_field = control_input
    elif isinstance(control_input, np.ndarray):
        control_field = lambda t: np.interp(t, times, control_input)
        # control_field = interp1d(times, control_input, kind='linear', fill_value='extrapolate')
        #control_field = lambda t : control_field_pre(t)
    else:
        control_field = lambda t: 0
        print("Control field's type not supported, set it to 0.")

    # pol_H = pol_overlaps["H"]
    # pol_V = pol_overlaps["V"]
    # def control_fun_H(t):
    #     return pol_H*control_field(t)
    # def control_fun_V(t):
    #     return pol_V*control_field(t)
    
    # complete control Hamiltonian and add to static Hamiltonian
    H = [H_0, [H_c,control_field]]
    #H = H_0

    # Define collapse operators for Lindblad master equation
    collapse_operators = [
        np.sqrt(gamma_e) * (ground_state * exciton_x.dag()),
        np.sqrt(gamma_e) * (ground_state * exciton_y.dag()),
        np.sqrt(gamma_b) * (exciton_x * biexciton.dag()),
        np.sqrt(gamma_b) * (exciton_y * biexciton.dag()),
        np.sqrt(gamma_d) * (ground_state * dark_exciton_x.dag()),
        np.sqrt(gamma_d) * (ground_state * dark_exciton_y.dag())
    ]

    # Define the initial state
    match rho_0_choice:
        case "G":
            rho0 = ground_state * ground_state.dag()
        case "X_H":
            rho0 = exciton_x * exciton_x.dag()
        case "X_V":
            rho0 = exciton_y * exciton_y.dag()
        case "D_H":
            rho0 = dark_exciton_x * dark_exciton_x.dag()
        case "D_V":
            rho0 = dark_exciton_y * dark_exciton_y.dag()
        case "B":
            rho0 = biexciton * biexciton.dag()

    # Define population operators for each state
    population_ops = [basis(N, i) * basis(N, i).dag() for i in range(N)]

    # Run the simulation using mesolve and calculate populations
    result = mesolve(H, rho0, times, collapse_operators, population_ops)

    return result


def plot_results_and_control(results,control_fun,t_array):

     # Plot population trajectories
    plt.figure(figsize=(10, 6))
    state_labels = ["|G>", "|X_H>", "|X_V>", "|D_H>", "|D_V>", "|B>"]
    for i in range(6):
        plt.plot(results.times, results.expect[i], label=(f"|{i}> = "+state_labels[i]))
    
    plt.xlabel("Time (ps)")
    plt.ylabel("Population")
    plt.title("Population Trajectories")
    plt.legend()
    plt.grid()
    plt.show()

    # visualize the control field
    control_FF_array = control_fun(t_array)
    plt.figure()
    plt.plot(t_array,control_FF_array )
    plt.xlabel("Time (ps)")
    plt.ylabel("Control Field (meV)")
    plt.title("Control Field")
    plt.show()

    control_FF_FFT = np.fft.fft(control_FF_array)
    control_FF_FFT = np.abs(control_FF_FFT)
    control_FF_FFT = control_FF_FFT[:len(control_FF_FFT)//2]
    plt.figure()
    plt.plot(control_FF_FFT)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Control Field FFT")
    plt.show()


if __name__ == "__main__":

    #load parameters
    par_QD = get_parameters()
    #polarization overlaps (e_H * e_L, e_V * e_L)
    polarization_overlaps = {"H": 1, "V":0}
    #choose initial state (G, X_H, X_V, D_H, D_V, B)
    init_state = "B"

    # Define time array for the simulation
    t_start = 0
    t_end = 1000  # ps
    dt = 0.1      # time step in ps
    t_array = np.linspace(t_start, t_end, int((t_end - t_start) / dt) + 1)

    #define control input
    # control_FF = lambda t: 10*np.sin(2*np.pi*t*1e-2)
    control_FF = lambda t: 100*np.exp(-t/4)*np.sin(2*np.pi*t)
    #control_FF = 1e-12*np.sin(2*np.pi*t_array*1e-3)

    simulation_result = simulate_dark_states(t_array,control_FF,init_state,polarization_overlaps,par_QD)
    print(simulation_result)
     
    # visualize the results, control field and control field in frequency domain
    plot_results_and_control(simulation_result,control_FF,t_array)






