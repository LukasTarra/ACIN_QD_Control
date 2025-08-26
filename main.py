import numpy as np
from qutip import basis, Qobj, mesolve
import matplotlib.pyplot as plt

from getparameters import get_parameters

def simulate_quantum_dot():
    # Load parameters from the params struct
    params = get_parameters()
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
    temperature = params.temperature

    # Define the six-level system

    N = 6  # Number of states
    ground_state = basis(N, 0)
    exciton_x = basis(N, 1)
    exciton_y = basis(N, 2)
    dark_exciton_x = basis(N, 3)
    dark_exciton_y = basis(N, 4)
    biexciton = basis(N, 5)

    # Define the Hamiltonian terms
    H0 = E_X_H * exciton_x * exciton_x.dag() + E_X_V * exciton_y * exciton_y.dag() + \
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
    H = H0 + H_bx + H_bz

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
    rho0 = biexciton * biexciton.dag()

    # Define time array for the simulation
    t_start = 0
    t_end = 1000  # ps
    dt = 0.5      # time step in ps
    times = np.linspace(t_start, t_end, int((t_end - t_start) / dt) + 1)

    # Define population operators for each state
    population_ops = [basis(N, i) * basis(N, i).dag() for i in range(N)]

    # Run the simulation using mesolve and calculate populations
    result = mesolve(H, rho0, times, collapse_operators, population_ops)

    return result

# Example usage
if __name__ == "__main__":
    simulation_result = simulate_quantum_dot()
    print(simulation_result)

    # Plot population trajectories
    plt.figure(figsize=(10, 6))
    state_labels = ["|G>", "|X_H>", "|X_V>", "|D_H>", "|D_V>", "|B>"]
    for i in range(6):
        plt.plot(simulation_result.times, simulation_result.expect[i], label=(f"|{i}> = "+state_labels[i]))
    
    plt.xlabel("Time (ps)")
    plt.ylabel("Population")
    plt.title("Population Trajectories")
    plt.legend()
    plt.grid()
    plt.show()