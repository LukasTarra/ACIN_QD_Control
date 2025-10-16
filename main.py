# import packages
import numpy as np
from qutip import basis, Qobj, mesolve
import matplotlib.pyplot as plt

# user-defined functions & packages
from getparameters import get_parameters

# Helper functions
def create_hamiltonian_terms(params):
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

    return H_QD, H_bx, H_bz, ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton


def create_control_field(times, control_input):
    zero_function = lambda t: 0
    if callable(control_input):
        control_field = control_input
    elif isinstance(control_input, np.ndarray):
        control_field = lambda t: np.interp(t, times, control_input)
    else:
        control_field = zero_function
        print("Control field's type not supported, set it to 0.")

    return control_field


def create_population_operators(N):
    return [basis(N, i) * basis(N, i).dag() for i in range(N)]


def create_collapse_operators(ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton, params):
    # Define decay rates for each state
    exciton_decay_rate = 1 / params.Gamma_X_inv  # Exciton decay rate
    dark_state_decay_rate = 0  # Decay rate for dark states
    biexciton_decay_rate = 1 / params.Gamma_XX_inv  # Biexciton decay rate

    # Create collapse operators with their respective decay rates
    collapse_ops = [
        np.sqrt(exciton_decay_rate) * (ground_state * exciton_x.dag()),
        np.sqrt(exciton_decay_rate) * (ground_state * exciton_y.dag()),
        np.sqrt(dark_state_decay_rate) * (ground_state * dark_exciton_x.dag()),
        np.sqrt(dark_state_decay_rate) * (ground_state * dark_exciton_y.dag()),
        np.sqrt(biexciton_decay_rate) * (exciton_x * biexciton.dag()),
        np.sqrt(biexciton_decay_rate) * (exciton_y * biexciton.dag())
    ]
    return collapse_ops


def create_initial_state(rho_0_choice, ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton):
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

    print("The chosen ground state is: ", rho_0_choice)

    return rho0


def simulate_dark_states(times, control_input, rho_0_choice, pol_overlaps, params):
    # Create Hamiltonian terms
    H_QD, H_bx, H_bz, ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton = create_hamiltonian_terms(params)

    # Total Hamiltonian (without control fields)
    H_0 = (H_QD + H_bx + H_bz) / params.hbar / 10  # Normalize by hbar

    # Add control Hamiltonians (factor for polarization overlap included)
    H_c_H = pol_overlaps["H"] * params.hbar * (exciton_x * ground_state.dag() + ground_state * exciton_x.dag() + exciton_x * biexciton.dag() + biexciton * exciton_x.dag())
    H_c_V = pol_overlaps["V"] * params.hbar * (exciton_y * ground_state.dag() + ground_state * exciton_y.dag() + exciton_y * biexciton.dag() + biexciton * exciton_y.dag())
    H_c = (H_c_H + H_c_V) / params.hbar # Normalize by hbar

    # Create control field
    control_field = create_control_field(times, control_input)

    # Complete control Hamiltonian and add to static Hamiltonian
    if control_field is None:
        H = H_0
        print("No control field provided, set it to 0.")
    elif control_field == (lambda t: 0):
        H = H_0
    else:
        H = [H_0, [H_c, control_field]]

    # Define population operators for each state
    N = 6  # Number of states
    population_ops = create_population_operators(N)

    # Create collapse operators
    collapse_operators = create_collapse_operators(ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton, params)

    # Create initial state
    rho0 = create_initial_state(rho_0_choice, ground_state, exciton_x, exciton_y, dark_exciton_x, dark_exciton_y, biexciton)

    # Run the simulation using mesolve and calculate populations
    result = mesolve(H , rho0, times, collapse_operators, population_ops)

    return result


def plot_population_trajectories(results):
    plt.figure(figsize=(10, 6))
    state_labels = ["|G>", "|X_H>", "|X_V>", "|D_H>", "|D_V>", "|B>"]
    for i in range(6):
        plt.plot(results.times, results.expect[i], label=(f"|{i}> = " + state_labels[i]))

    plt.xlabel("Time (ps)")
    plt.ylabel("Population")
    plt.title("Population Trajectories")
    plt.legend()
    plt.grid()
    plt.show()

def plot_control_field(control_fun, t_array):
    control_FF_array = control_fun(t_array)
    plt.figure()
    plt.plot(t_array, control_FF_array)
    plt.xlabel("Time (ps)")
    plt.ylabel("Control Field (meV)")
    plt.title("Control Field")
    plt.show()

def plot_control_field_fft(control_fun, t_array):
    control_FF_array = control_fun(t_array)
    control_FF_FFT = np.fft.fft(control_FF_array)
    control_FF_FFT = np.abs(control_FF_FFT)
    control_FF_FFT = control_FF_FFT[:len(control_FF_FFT)//2]
    plt.figure()
    plt.plot(control_FF_FFT)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Control Field FFT")
    plt.show()

def plot_results_and_control(results, control_fun, t_array):
    plot_population_trajectories(results)
    plot_control_field(control_fun, t_array)
    plot_control_field_fft(control_fun, t_array)

# set up a results class for diagnostics
class Results:
    def __init__(self, sim_result, params, control_field=None):
        self.times = sim_result.times
        self.populations = sim_result.expect
        self.params = params
        self.control_field = control_field

    def get_populations(self):
        return self.populations
    def get_control_field(self):
        return self.control_field
    def get_times(self):
        return self.times
    def get_params(self):
        return self.params

    def get_population_at_time(self, time_index):
        """Get population of all states at a specific time index."""
        if time_index < 0 or time_index >= len(self.times):
            raise ValueError("Time index out of range")
        return self.populations[time_index]
    def get_state_population(self, state_index):
        """Get population trajectory for a specific state."""
        if state_index < 0 or state_index >= len(self.populations[0]):
            raise ValueError("State index out of range")
        return [p[state_index] for p in self.populations]
    def get_final_populations(self):
        """Get the population of all states at the final time."""
        return self.populations[-1]
    def get_max_population(self, state_index):
        """Get the maximum population reached by a specific state."""
        state_pop = self.get_state_population(state_index)
        return max(state_pop)
    def calculate_fidelity(self, target_state):
        """Calculate fidelity with respect to a target state at the final time."""
        final_pop = self.get_final_populations()
        return final_pop[target_state]
    def calculate_purity(self):
        """Calculate purity of the system over time (trace of rho^2)."""
        purity = []
        for pop_matrix in self.populations:
            # For diagonal density matrix elements, purity is sum of squares
            purity.append(sum(p**2 for p in pop_matrix))
        return purity
    def save_to_file(self, filename):
        """Save results to a file for later analysis."""
        data = {
            'times': self.times,
            'populations': self.populations,
            'params': self.params.__dict__ if hasattr(self.params, '__dict__') else self.params,
            'control_field': self.control_field(self.times) if callable(self.control_field) else None
        }
        np.savez(filename, **data)
    def plot_state_populations(self, state_indices=None):
        """Plot population trajectories for specified states or all states."""
        plt.figure(figsize=(10, 6))
        state_labels = ["|G>", "|X_H>", "|X_V>", "|D_H>", "|D_V>", "|B>"]
        if state_indices is None:
            state_indices = range(len(self.populations[0]))
        for i in state_indices:
            plt.plot(self.times, self.get_state_population(i), label=state_labels[i])
        plt.xlabel("Time (ps)")
        plt.ylabel("Population")
        plt.title("Population Trajectories")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # Load parameters
    par_QD = get_parameters()
    # Polarization overlaps (e_H * e_L, e_V * e_L)
    polarization_overlaps = {"H": 1, "V": 0}
    # Choose initial state (G, X_H, X_V, D_H, D_V, B)
    init_state = "B"

    # Define time array for the simulation
    t_start = 0
    t_end = 1000  # ps
    dt = 0.05      # time step in ps
    t_array = np.linspace(t_start, t_end, int((t_end - t_start) / dt) + 1)

    # Define control input
    control_FF = lambda t: 1 * (1/(1+ t**2/100)) * np.sin(2 * np.pi * t)
    plot_control_field(control_FF, t_array)
    plot_control_field_fft( control_FF, t_array )
    # carry out the simulation
    simulation_result = simulate_dark_states(t_array, control_FF, init_state, polarization_overlaps, par_QD)
    # Visualize the results
    plot_population_trajectories(simulation_result)

