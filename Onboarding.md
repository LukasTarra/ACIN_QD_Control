This document explains the physical model, its mathematical formulation, and the\
rotating-frame transformation implemented in `qdot_simulation.py`. It is\
intended as a self-contained guide for onboarding a new master's student to the\
project.

The derivations of the rotating-frame transformation are taken from the\
companion Maple script `Rotating_frame_calculations.mw` (the equation numbers\
referenced throughout correspond to the Maple output labels (1)–(16)).

---

## 0. Project Structure

### 0.1 Codebase

The codebase consists of four main Python modules:

| File                 | Purpose                                                                                                                                                                                            |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `getparameters.py`   | Defines simulation and pulse parameters (`ParametersQD`, `DefaultPulse` classes). Central configuration point for physical constants, QD properties, and pulse shapes.                             |
| `utilities.py`       | Helper functions: `Timer` class for profiling, complex-to-real block matrix conversions (`complex_to_real_block()`, `real_to_complex_block()`, etc.). Used throughout for JAX compatibility.       |
| `qdot_simulation.py` | Core simulation engine: Hamiltonian construction, rotating-frame transformation, collapse operators, SSE integrator (`em_step()`), trajectory batching, and result visualization (`QDSimResults`). |
| `qdot_control.py`    | High-level control and optimization: cost functions, pulse parameter optimization (`optimize_delta_p()`), and main simulation workflow (`main()`).                                                 |

### 0.2 Call Stack

Understanding which files call which is essential for navigating the codebase.\
The dependency structure is:

```
qdot_control.py (entry point)
    ├── imports: getparameters.py (ParametersQD, DefaultPulse)
    ├── imports: utilities.py (Timer)
    ├── imports: qdot_simulation.py (jax_sim_setup, simulate_batch, QDSimResults)
    │
    └── main()
        ├── get_parameters_QD() → ParametersQD instance
        ├── jax_sim_setup() → pre-computed Hamiltonians, operators, initial state
        ├── get_default_pulse() → DefaultPulse instances (initialization, storage, retrieval)
        ├── optimize_delta_p() → optimizes pulse parameters
        │   └── calls: get_chirped_pulse_function(), simulate_batch(), cost_function_target_state()
        ├── simulate_batch() → runs ensemble trajectories
        └── QDSimResults.plot_*() → visualization

qdot_simulation.py
    ├── imports: getparameters.py (ParametersQD)
    ├── imports: utilities.py (complex_to_real_block, complex_to_real_vector, normalize_psi)
    │
    └── Key functions:
        ├── create_complex_QD_state_vectors() → basis kets
        ├── create_QD_hamiltonian_terms_and_states() → H_0
        ├── create_dressed_energies_and_states() → analytical diagonalization
        ├── create_collapse_operators() → Lindblad operators
        ├── transform_Hamiltonian_to_rotating_frame() → H_0_tilde_eff
        ├── create_control_Hamiltonians_rotating() → H_c_tilde_real/imag
        ├── jax_sim_setup() → assembles all simulation matrices
        ├── em_step() → single SSE integration step
        ├── simulate_single_traj() → one trajectory
        └── simulate_batch() → vmap'd ensemble

getparameters.py
    └── Standalone parameter definitions (no internal dependencies)

utilities.py
    └── Standalone utilities (no internal dependencies)
```

**Execution flow:**

1. `qdot_control.py` is the entry point (run with `python qdot_control.py`)
2. Parameters loaded from `getparameters.py` via `get_parameters_QD()`
3. Simulation matrices assembled in `qdot_simulation.py` via `jax_sim_setup()`
4. Optimization loop in `optimize_delta_p()` modifies pulse parameters and calls `simulate_batch()`
5. Results visualized via `QDSimResults` methods

**Key insight:** `qdot_simulation.py` is the computational core (physics + numerics), while `qdot_control.py` orchestrates the workflow and adds optimization layers.

---

## 1. Physical System

We model a single semiconductor quantum dot (QD) that is driven by a pulsed laser\
field and is subject to an external static magnetic field. Six states are kept\
explicitly in the simulation:

| Index | State                 | Physical meaning                                           |
| ----- | --------------------- | ---------------------------------------------------------- |
| 0     | $$|G\rangle$$         | Crystal ground state (no electron–hole pair)               |
| 1     | $$\lvert X_H\rangle$$ | Bright exciton, H-polarised (heavy-hole, $$\pm 1$$ valley) |
| 2     | $$\lvert X_V\rangle$$ | Bright exciton, V-polarised                                |
| 3     | $$\lvert D_H\rangle$$ | Dark exciton, H-valley                                     |
| 4     | $$\lvert D_V\rangle$$ | Dark exciton, V-valley                                     |
| 5     | $$\lvert B\rangle = \lvert B\rangle$$   | Biexciton (two electron–hole pairs)      |

The bright–dark pairs are coupled through the in-plane component $$B_x$$ of the\
magnetic field (fine-structure mixing). The on-axis component $$B_z$$ induces a\
Zeeman splitting between the H and V components within the bright and the dark\
manifolds respectively.

Phonon coupling is **not** included in this model; the `pyaceqd` package is the\
external tool that would be used to incorporate phonon processes if needed.

The Hilbert-space dimension is $$N = 6$$. The state vector\
$$\lvert \psi(t)\rangle \in \mathbb{C}^6$$ evolves under a non-Hermitian\
stochastic Schrödinger equation (quantum-jump / Monte-Carlo wave-function\
approach).

---

## 2. Mathematical Description

### 2.1 Hamiltonian

The total Hamiltonian in the Schrödinger picture is

```math
\hat H(t) \;=\; \hat H_{\text{QD}} + \hat H_{B_x} + \hat H_{B_z} + \hat H_L(t) ,
```

- $$\hat H_{\text{QD}}$$ — diagonal energy operator,
- $$\hat H_{B_x}$$ — bright–dark mixing from the transverse magnetic field,
- $$\hat H_{B_z}$$ — on-axis Zeeman splitting,
- $$\hat H_L$$ — laser control Hamiltonian.

#### 2.1.1 Diagonal part $\hat H\_{\text{QD}}$

```math
\hat H_{\text{QD}} \;=\; E_{X_H}\, \lvert X_H\rangle\!\langle X_H\rvert
+E_{X_V}\, \lvert X_V\rangle\!\langle X_V\rvert
+E_{D_H}\, \lvert D_H\rangle\!\langle D_H\rvert
+E_{D_V}\, \lvert D_V\rangle\!\langle D_V\rvert
+E_B\, \lvert B\rangle\!\langle B\rvert .
```

(Eq. (4) in `Rotating_frame_calculations.mw`.) This is implemented in`create_QD_hamiltonian_terms_and_states()` by building each term as\
$$E_i,\lvert i\rangle \langle i\rvert = E_i,\mathbf{e}_i,\mathbf{e}_i^\dagger$$.

#### 2.1.2 Bright–dark coupling $\hat H\_{B\_x}$

For an in-plane field $$B_x$$ the exciton and the dark-exciton of the **same valley** are coupled. Defining

```math
j^+ \;=\; -\tfrac{1}{2}\, \mu_B\, B_x\,(g_{h_x}+g_{e_x}), \qquad
j^- \;=\; -\tfrac{1}{2}\, \mu_B\, B_x\,(g_{h_x}-g_{e_x}),
```

we have

```math
\hat H_{B_x} \;=\; j^+\bigl(\lvert X_H\rangle\!\langle D_H\rvert +
\lvert D_H\rangle\!\langle X_H\rvert\bigr)
+j^-\bigl(\lvert X_V\rangle\!\langle D_V\rvert +
\lvert D_V\rangle\!\langle X_V\rvert\bigr) .
```

In the code (function `create_QD_hamiltonian_terms_and_states`):

```python
j_plus  = -0.5 * mu_b * bx * (g_hx + g_ex)
j_minus = -0.5 * mu_b * bx * (g_hx - g_ex)
```

#### 2.1.3 On-axis Zeeman term $\hat H\_{B\_z}$

```math
\hat H_{B_z} = \tfrac{i}{2}\mu_B B_z (g_{e_z}-3g_{h_z})
\bigl(\lvert X_H\rangle\!\langle X_V\rvert -
\lvert X_V\rangle\!\langle X_H\rvert\bigr)
-\tfrac{i}{2}\mu_B B_z (g_{e_z}+3g_{h_z})
\bigl(\lvert D_H\rangle\!\langle D_V\rvert -
\lvert D_V\rangle\!\langle D_H\rvert\bigr) .
```

In the current code, $$g_{e_z}=g_{h_z}=0$$, so this term vanishes. The structure\
is nevertheless coded (with `bz != 0` as a guard) for future use.

#### 2.1.4 Laser control $\hat H\_L$

The control Hamiltonian is

```math
\hat H_L(t) \;=\; \Omega(t)\,
\bigl[e_H (\lvert X_H\rangle\!\langle G\rvert+\lvert B\rangle\!\langle X_H\rvert)
+e_V (\lvert X_V\rangle\!\langle G\rvert+\lvert B\rangle\!\langle X_V\rvert)\bigr]
\;+\;\Omega^*(t)\,[\text{h.c.}],
```

where $$e_H,,e_V\in\mathbb{C}$$ are the polarisation overlaps of the laser with\
the two dipole-allowed transitions, and $$\Omega(t)$$ is the (user-supplied)\
complex pulse envelope. (Eq. (7) in `Rotating_frame_calculations.mw`.)

In the Schrödinger picture $$\Omega(t)$$ contains a fast oscillation at the\
carrier frequency $$\omega_p \approx \omega_X$$. The rotating-frame transformation\
in §3 removes this oscillation, leaving only the slowly-varying envelope.

### 2.2 Open-system dynamics — Stochastic Schrödinger Equation (SSE)

The simulation propagates an **ensemble** of state vectors\
$${\lvert\psi(t)\rangle}$$ using the Itô stochastic Schrödinger equation (quantum-jump / Monte-Carlo wave-function form):

```math
\boxed{\;
\mathrm d\psi \;=\; -\tfrac{i}{\hbar}\,\hat H\,\psi\,\mathrm dt
\;-\;\tfrac{1}{2}\!\sum_k\bigl(\hat L_k^\dagger \hat L_k
- 2\,c_k\,\hat L_k + c_k^2\bigr)\psi\,\mathrm dt
\;+\;\sum_k\bigl(\hat L_k - c_k\bigr)\psi\,\mathrm dW_k
\;}
```

with

```math
c_k \;\equiv\; \tfrac{1}{2}\langle\psi\rvert(\hat L_k^\dagger+\hat L_k)\lvert\psi\rangle ,
\qquad
\langle\mathrm dW_k\,\mathrm dW_{k'}\rangle = \delta_{kk'}\,\mathrm dt,\qquad
\langle\mathrm dW_k\rangle = 0 .
```

#### Why SSE and not the Lindblad master equation?

The Lindblad equation evolves the $$6\times 6$$ **density matrix** $$\hat\rho(t)$$\
which has $$72$$ real degrees of freedom. The SSE, by contrast, evolves only a\
single complex state vector ($$12$$ real degrees of freedom) per trajectory.\
Both formulations are equivalent for the ensemble average,

```math
\langle\hat\rho(t)\rangle \;=\; \tfrac{1}{N_{\text{traj}}}\sum_{r=1}^{N_{\text{traj}}}
\lvert\psi^{(r)}(t)\rangle\!\langle\psi^{(r)}(t)\rvert ,
```

but the SSE allows every trajectory to be propagated **independently** in\
parallel. Because the noise terms $$\mathrm dW_k^{(r)}$$ are uncorrelated across\
trajectories, the loop over $$r$$ parallelises trivially — in JAX this is done\
with a single `vmap`, which executes each trajectory on a separate GPU thread\
or CPU core. In practice, simulating $$N_{\text{traj}}\sim10^2-10^3$$\
trajectories on a GPU is orders of magnitude faster than the equivalent\
Lindblad propagation of a single $$72$$-dimensional real vector.

### 2.3 Collapse (jump) operators

Spontaneous emission is included with four Lindblad operators (Eq. (12) in`Rotating_frame_calculations.mw`, and `create_collapse_operators()`):

```math
\hat L_1 = \sqrt{\Gamma_X}\,\lvert G\rangle\!\langle X_H\rvert,\qquad
\hat L_2 = \sqrt{\Gamma_X}\,\lvert G\rangle\!\langle X_V\rvert,
```

```math
\hat L_3 = \sqrt{\Gamma_{XX}}\,\lvert X_H\rangle\!\langle B\rvert,\qquad
\hat L_4 = \sqrt{\Gamma_{XX}}\,\lvert X_V\rangle\!\langle B\rvert .
```

Dark-state radiative decay is **switched off** in the code (commented out) to\
reduce stochastic noise.

### 2.4 Two-photon resonance condition

The carrier frequency is chosen close to the *two-photon* resonance of the\
biexciton:

```math
\hbar\,\omega_p \;\approx\; E_{XX}/2 \;=\; \hbar\,\omega_X - E_B/2 ,
```

i.e. the detuning from the single-exciton transition is $$E_B/2$$ (the biexciton\
binding energy divided by 2). Under this condition the pulse drives the\
$$\lvert G\rangle\leftrightarrow\lvert B\rangle$$ transition directly via two\
virtual single-exciton intermediates, while keeping the populations of\
$$\lvert X_{H/V}\rangle$$ small. This is the key idea behind protocols that\
suppress unwanted excitation of the bright exciton and channel the population\
into the dark-exciton subspace.

---

## 3. Rotating-Frame Transformation — the Key Simplification

This is the central trick that lets us integrate the SSE with a much larger\
time-step. The derivation is taken directly from `Rotating_frame_calculations.mw`.

### 3.1 Motivation

The laser carrier frequency is $$\omega_p \approx \omega_X \sim 10^{3},\text{ps}^{-1}$$.\
To resolve this oscillation explicitly we would need $$\Delta t \ll 1,\text{fs}$$.\
By moving into a frame that rotates at this frequency, the rapidly oscillating\
exponentials are removed and the dynamics of the slow envelope $$\Omega(t)$$ is\
what remains.

### 3.2 The unitary $U(t)$ (Eq. (1))

Define

```math
\hat U(t) \;=\; \text{diag}\!\bigl(1,\,e^{-i\omega t},\,e^{-i\omega t},
\,e^{-i\omega t},\,e^{-i\omega t},\,e^{-i\,2\,\omega t}\bigr),
```

where $$\omega = E_X/\hbar$$ is the bare exciton frequency. The last entry is\
multiplied by $$2$$ because the biexciton carries **two** excitons — its energy\
$$E_B \approx 2E_X$$. The corresponding generator (Eq. (2))

```math
\hat V \;=\; \frac{1}{\omega}\,\dot{\hat U}(0) \;=\;
\text{diag}(0,1,1,1,1,2)
```

is the **excitation-counting operator** (the number of electron–hole pairs in\
each basis state).

### 3.3 Transforming the state

Set

```math
\lvert \tilde\psi(t)\rangle \;=\; \hat U^\dagger(t)\,\lvert\psi(t)\rangle
\qquad\Longleftrightarrow\qquad
\lvert\psi(t)\rangle = \hat U(t)\,\lvert\tilde\psi(t)\rangle .
```

Differentiating,

```math
\mathrm d\psi \;=\; \mathrm d\hat U\,\tilde\psi + \hat U\,\mathrm d\tilde\psi,
\qquad
\mathrm d\hat U = -i\omega\,\hat U\,\hat V\,\mathrm dt ,
```

plugging into the SSE and multiplying from the left by $$\hat U^\dagger$$ gives\
the transformed SSE

```math
\mathrm d\tilde\psi \;=\; -\tfrac{i}{\hbar}\,\tilde{\hat H}\,\tilde\psi\,\mathrm dt
\;-\;\tfrac{1}{2}\!\sum_k\!\bigl(\tilde L_k^\dagger\tilde L_k
-2\tilde c_k\tilde L_k+\tilde c_k^2\bigr)\tilde\psi\,\mathrm dt
\;+\;\sum_k\!\bigl(\tilde L_k-\tilde c_k\bigr)\tilde\psi\,\mathrm dW_k ,
```

with

```math
\tilde{\hat H} = \hat U^\dagger\hat H\,\hat U + \hbar\omega\,\hat V,\qquad
\tilde L_k = \hat U^\dagger L_k\,\hat U,\qquad
\tilde c_k = \tfrac12\langle\tilde\psi\rvert(\tilde L_k^\dagger+\tilde L_k)
\lvert\tilde\psi\rangle .
```

### 3.4 What the transformation does to the Hamiltonians

#### 3.4.1 Free Hamiltonian (Eq. (4) → (6))

The free Hamiltonian $$\hat H_0 = \hat H_{\text{QD}}+\hat H_{B_x}+\hat H_{B_z}$$\
contains the diagonal energies and the bright–dark couplings. The latter\
connect states with the **same** excitation number (a bright and a dark\
exciton of the same valley) and therefore commute with $$\hat V$$. Consequently\
$$\hat U^\dagger\hat H_0,\hat U = \hat H_0$$ and (Eq. (6))

```math
\tilde{\hat H}_0 \;=\; \hat H_0 \;-\;\hbar\omega\,
\text{diag}(0,1,1,1,1,2).
```

This is exactly what the code computes in`transform_Hamiltonian_to_rotating_frame`:

```python
rotating_frame_diagonal = jnp.array([0, 1, 1, 1, 1, 2])
H_0_tilde_eff = H_0 - hbar_omega_X * jnp.diag(rotating_frame_diagonal)
```

All single-exciton energies are now shifted by $-\hbar\omega$, leaving only *detunings* (fine-structure splittings $$\delta_X = E_{X_H}-E_{X_V}$$ and the\
bright–dark splittings $$E_{X_{H/V}}-E_{D_{H/V}}$$) in the diagonal. The\
biexciton row is shifted by $$-2\hbar\omega$$, leaving the binding energy $$E_B$$\
as its only contribution.

#### 3.4.2 Control Hamiltonian (Eq. (7) → (9))

In the Schrödinger picture $$\Omega(t)$$ is rapidly oscillating at the carrier\
frequency. After the transformation only the slow envelope remains. Following\
the Maple derivation (Eqs. (8) and (9)):

```math
\tilde{\hat H}_L(t) \;=\;
\tilde\Omega(t)\,\tilde{\hat H}_c^{(1)}
\;+\; \tilde\Omega^*(t)\,\tilde{\hat H}_c^{(2)} ,
```

where $$\tilde\Omega(t) \equiv \Omega(t),e^{i\omega_p t}$$ (a slowly-varying envelope when $$\omega_p \approx \omega$$) and

```math
\tilde{\hat H}_c^{(1)} \;=\; e_H\bigl(\lvert X_H\rangle\!\langle G\rvert
+\lvert B\rangle\!\langle X_H\rvert\bigr)
+e_V\bigl(\lvert X_V\rangle\!\langle G\rvert
+\lvert B\rangle\!\langle X_V\rvert\bigr),
```

with $$\tilde{\hat H}_c^{(2)} = (\tilde{\hat H}_c^{(1)})^\dagger$$.

Splitting $$\tilde\Omega = \Omega_R + i,\Omega_I$$ and using\
$$\tilde\Omega^* = \Omega_R - i,\Omega_I$$ one obtains (after simplification,\
using the $$A$$ and $$B$$ matrices of Eqs. (10)–(11))

```math
\tilde{\hat H}_L \;=\; \Omega_R\,\bigl(\tilde{\hat H}_c^{(1)}+\tilde{\hat H}_c^{(2)}\bigr)
\;+\; \Omega_I\,\bigl(i\tilde{\hat H}_c^{(1)}-i\tilde{\hat H}_c^{(2)}\bigr),
```

which is implemented in `create_control_Hamiltonians_rotating()`:

```python
H_c_tilde_real = H_c_1 + H_c_2
H_c_tilde_imag = 1j*H_c_1 - 1j*H_c_2
```

This separation is what allows `em_step` to multiply a **time-independent**\
matrix by a **time-dependent scalar** (`Omega_real`, `Omega_imag`) at every\
step.

#### 3.4.3 Collapse operators (Eq. (12) → (16))

Each $$\hat L_k$$ lowers the excitation number by exactly one, so\
$$\hat U^\dagger\hat L_k,\hat U = e^{i\omega t}\hat L_k$$. Therefore (Eq. (13))

```math
\tilde L_k \;=\; e^{-i\omega t}\,\hat L_k,\qquad
\tilde L_k^\dagger\tilde L_k = \hat L_k^\dagger\hat L_k ,
```

i.e. the **rate terms**$$\tilde L_k^\dagger\tilde L_k$$are unchanged, while\
the stochastic part picks up an overall phase $$e^{-i\omega t}$$. In practice\
these phases only enter the diffusion (stochastic) part, which is already\
stochastic, and they may be absorbed into the Wiener increments without\
changing the statistics of the ensemble. The code therefore uses the *original-frame*  collapse operators in the diffusion term; only the diagonal\
(drift) part is genuinely affected by the rotation, and that part is\
unchanged.

### 3.5 Dressed-state picture (analytical baseline)

Before running the simulation, `create_dressed_energies_and_states()`\
analytically diagonalises the bright–dark $$2\times 2$$ blocks of $$\hat H_0$$\
in each polarisation:

```math
E_{X_{H/V}}^{\pm} \;=\; \tfrac12\!\Bigl(\sqrt{(E_{X_{H/V}}-E_{D_{H/V}})^2+4(j^{\pm})^2}
\;+\;E_{X_{H/V}}+E_{D_{H/V}}\Bigr),
```

```math
\lvert X_{H/V}^{\pm}\rangle \;=\;
\mathcal N\!\Bigl[\bigl(\tfrac{\Delta+\sqrt{\Delta^2+4(j^{\pm})^2}}{2j^{\pm}}\bigr)
\lvert X_{H/V}\rangle \;+\; \lvert D_{H/V}\rangle\Bigr],
```

with $$\Delta = E_{X_{H/V}}-E_{D_{H/V}}$$. When $$j^{\pm}=0$$ (no in-plane field)\
the dressed energies reduce to the bare energies $$E_{X_{H/V}}, E_{D_{H/V}}$$.

These dressed states are used only for *diagnostic* purposes (e.g.`plot_mean_dressed_population_trajectories()`); the propagation always stays\
in the bare basis.

---

## 4. Numerical Integration

### 4.1 Euler–Maruyama with sub-stepping

`em_step()` integrates one big time-step $$\Delta t$$ as follows.

1. **Deterministic part (RK2).** Split the step into $$N_{\text{Ham}}$$ sub-steps\
   and integrate the Hamiltonian piece with a 2nd-order Runge–Kutta (RK2 /\
   midpoint) scheme:
   ```python
   def hamiltonian_rk2_step(psi, I_H_total, dt_step):
       k1 = -dt_step * I_H_total @ psi
       psi_mid = psi + 0.5 * k1
       k2 = -dt_step * I_H_total @ psi_mid
       return psi + k2
   ```
   The use of `jax.lax.scan` (rather than `fori_loop`) keeps the sub-stepping **differentiable**, which is useful if one later wishes to backpropagate\
   through the dynamics for optimal-control purposes.
2. **Diffusion part (Itô correction).** Compute the Lindblad expectations\
   $$c_k$$ and the diffusion coefficients $$L_k - c_k\mathbb{1}$$ (in the\
   real-block representation), then advance
   ```math
   \psi \;\mapsto\; \psi + \Delta t\,D[\psi]
   \;+\;\sum_k(L_k - c_k\mathbb{1})\psi\,\Delta W_k ,
   ```
   where $D\[\psi]$ is the deterministic Lindblad drift:
   ```math
   D[\psi] = -\tfrac12\sum_k L_k^\dagger L_k\psi
   \;+\;\sum_k\langle L_k\rangle L_k\psi
   \;-\;\tfrac12\sum_k\langle L_k\rangle\langle L_k\rangle\psi .
   ```
3. **Renormalisation.** After every step `normalize_psi()` projects the state\
   onto the unit sphere. A small regularisation ($$10^{-14}$$) prevents\
   division-by-zero if a jump takes $$\lVert\psi\rVert$$ to (almost) zero.

### 4.2 Real block representation (Mathematical Appendix)

Because `JAX.trace` and `lax.scan` are much faster on real-valued code, every\
complex matrix/vector is mapped to a $$2N\times 2N$$ real block form by the\
helpers `complex_to_real_block()` and `complex_to_real_vector()`. For a\
complex matrix $$A = A_r + i A_i$$:

```math
A \;\longmapsto\; \begin{pmatrix} A_r & -A_i \\ A_i & A_r \end{pmatrix} ,
\qquad
\Psi \;\longmapsto\; \begin{pmatrix} \text{Re}\,\Psi \\ \text{Im}\,\Psi \end{pmatrix} .
```

The action of the imaginary unit $$i$$ becomes the canonical $$12\times 12$$ matrix

```math
I_{\text{imag}} \;=\; \begin{pmatrix} 0 & -\mathbb{1}\\ \mathbb{1} & 0\end{pmatrix},
```

so that $$H\psi \to (I_{\text{imag}}\cdot H),\psi_{\text{real}}$$. This is why\
all the products `I_H_0_tilde_eff_j`, `I_H_c_tilde_real_j`, `I_H_c_tilde_imag_j`\
in `jax_sim_setup()` are pre-computed once before the time loop.

### 4.3 Batching trajectories

The single-trajectory simulator `simulate_single_traj()` is wrapped with`jax.vmap` over the leading batch axis of the noise arrays, giving`sim_forward_vmap`. This makes the parallel cost essentially free in JAX.

The complete ensemble is therefore produced by one JIT-compiled call per\
pulse-shape update, which is the design point of the code: changing the\
control field does **not** require recompilation.

---

## 5. Walk-through of qdot\_simulation.py

The file is organised as a pipeline. The recommended reading order for a new\
student is:

1. **`create_complex_QD_state_vectors()`** — defines the six basis kets.
2. **`create_QD_hamiltonian_terms_and_states()`** — implements §2.1.1–2.1.3.
3. **`create_dressed_energies_and_states()`** — analytical diagonalisation of\
   the bright–dark $$2\times 2$$ blocks (§3.5).
4. **`create_collapse_operators()`** — implements §2.3.
5. **`transform_Hamiltonian_to_rotating_frame()`** — implements §3.4.1\
   ("Hamiltonian" part).
6. **`create_control_Hamiltonians_rotating()`** — implements the\
   $$\Omega_R$$/$$\Omega_I$$ splitting of §3.4.2 ("control" part).
7. **`jax_sim_setup()`** — pulls everything together, builds the real-block\
   matrices, pre-multiplies by $$I_{\text{imag}}$$ and returns the arrays used\
   inside the loop.
8. **`create_jax_noise_traj_arrays()`** — draws the Wiener increments\
   $$\Delta W_k$$.
9. **`em_step()`** **/** **`simulate_single_traj()`** — the Euler–Maruyama/RK2\
   integrator (§4.1).
10. **`simulate_batch()`** — `vmap`'d batch driver (§4.3).
11. **`QDSimResults`** — convenience class for plotting mean and individual\
    trajectories (in the bare or dressed basis) plus diagnostics on the\
    control field (real/imaginary part and FFT).

---

## 6. Quick sanity checks

When onboarding, the following consistency checks are useful:

- **Diagonal shift.** With $$B_x=0$$ and $$B_z=0$$, $$\hat H_0$$ is diagonal and`transform_Hamiltonian_to_rotating_frame` should subtract exactly\
  $$\hbar\omega$$ from rows/columns 1–4 and $$2\hbar\omega$$ from row/column 5.
- **No-control dynamics.** Setting $\Omega(t)=0$ should give purely\
  exponential decay of $$\lvert X_{H/V}\rangle$$ with rate $$\Gamma_X$$ and\
  biexciton→exciton feeding; the dark states should remain stationary (their\
  decay channels are commented out).
- **Two-photon resonance.** With the biexciton binding $$E_B$$ positive,\
  $$\tilde H_L$$ should produce visible $$\lvert G\rangle!\leftrightarrow!\lvert B\rangle$$\
  Rabi oscillations while keeping the populations of $$\lvert X_H\rangle$$ and\
  $$\lvert X_V\rangle$$ small — that is the whole point of the rotating-frame /\
  off-resonant driving trick.
- **Dressed-state rotation.** With $$B_x\neq 0$$, the populations should look\
  qualitatively different in the bare basis versus the dressed basis\
  (`plot_mean_dressed_population_trajectories`).

---

## 7. Future work

### 7.1 Validate / fix the simulated behavior

I am still unsure whether the simulation is entirely physically correct. The general characteristics, decay rates etc. agree with the "Keeping the photon in the dark" paper, but the same three-pulse combination from the paper (initialization - storage - retrieval) doesn't work as well in my simulation as it does in the paper. It might just be that it needs higher temporal resolution, though.\
Nonetheless, it would be good to make sure first that the results match the paper before going on to optimal control.\
One thing I believe would make sense is to compare the current real-numbered represenations of vectors & matrices to the built-in complex numbers, which would make the conversion complex->real in the initialization of the code obsolete.\
It would be interesting to compare the speed and results of real & complex representations.\
Also, I always assumed that it's not necessary to transform the Wiener noise process to the rotating frame, while in reality,

```math
dW_k \rightarrow \exp({-i\omega t}) dW_k \ .
```

This should be checked.

### 7.2 Optimize predefined pulses

Once simulations are reliable, the obvious first step is to just statically optimize the parameters of the three predefined chirped pulses (initialization - storage - retrieval). One could optimize frequency, chirping, amplitude, timing etc.\
The goal in all optimizations would be to yield the maximum occupation of the dark state $$\lvert D_H\rangle$$ after applying the init & storage pulses, and then get the maximum amount of $$\lvert B\rangle$$ after applying the retrieval pulse from there.
So, for a desired state $$\lvert \psi_d \rangle$$, we can use the cost function
```math
\frac{c_\psi}{2}\langle \psi(T) \rvert \left( 1-\lvert \psi_d \rangle \langle \psi_d \rvert \right) \lvert \psi(T) \rangle + \frac{c_u}{2} \int_{-\infty}^{\infty} \lvert u(t) \rvert^2 \mathrm{d}t
```

### 7.3 Perform optimal feedforward control

Instead of optimizing predefined chirped pulses (based on a 4f experimental setup), the experimental setup can be adapted to allow for more general actuation.\
In the SLM+block setup, the input signal is spatially decomposed into its wavelength spectrum and sent through an SLM (an arbitrary filter $$H(\omega)$$, where the frequency spectrum is discretized due to the finite spatial dimension of each SLM pixel). Additionally, a literal small, black block is put at the laser wavelength (blocking the laser signal) where the output signal coming from the quantum dot sits spectrally. This block therefore disambiguates the control input from the system's output in measurement.

In a frame rotating at $\omega$, the input signal is $$\Omega(t) \exp(i \omega t)$$. Thus, in the frequency domain, the input signal reads as

```math
\Omega(\omega_{RF}) = \Omega(\omega_{lab}-\omega) \ ,
```

and with the SLM+block setup,

```math
\Omega(\omega_{RF}) = H(\omega_{RF}) u(\omega_{RF}) \ ,
```

where $$u(\omega_{RF})$$ is the real new input coming from the SLM, and $$H(\omega_{RF})$$ represents the selective spectral filtering of the block (probably via a notch filter). In other words, the block represents actuator dynamics which the input $$u$$ goes through before entering the SSE.\
To interface $$\Omega(\omega_{RF})$$ with the time-domain SSE, just send $$\Omega(\omega_{RF})$$ through a Fast Fourier Transform before sending it into the simulate\_batch() function.
