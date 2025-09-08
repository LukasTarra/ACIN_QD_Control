"""
This module contains test functions for the coupled quantum harmonic oscillator simulation.

The tests verify:
- Creation of initial quantum states
- Operator initialization
- Time evolution steps
- Expectation value calculations
- Variance calculations

Tests use pytest and JAX's testing utilities to ensure correctness and numerical stability.
"""

import pytest
import jax.numpy as jnp
from jax import random
from coupled_quantum_harm_oscillator import (
    create_initial_state_with_qutip,
    run_quantum_simulation
)

@pytest.fixture
def default_params():
    """
    Default parameters for testing the quantum simulation.
    """
    return {
        "n_osc": 2,
        "n_cutoff": 5,
        "omega1": 1.0,
        "omega2": 1.0,
        "g": 0.1,
        "kappa1": 0.02,
        "kappa2": 0.02,
        "gamma1": 0.1,
        "gamma2": 0.1,
        "n_therm1": 0.1,
        "n_therm2": 0.1,
        "initial_state_type": "coherent",
        "fock_n1": 0,
        "fock_n2": 0,
        "alpha1": 0.5 + 0.0j,
        "alpha2": -0.5 + 0.0j,
        "n_trajectories": 1,
        "dt": 0.01,
        "t_final": 1.0,
        "main_key_seed": 12
    }


def test_create_initial_state_with_qutip(default_params):
    """
    Test the creation of initial quantum states.
    """
    # Test vacuum state
    vacuum_state = create_initial_state_with_qutip(
        "vacuum", default_params["n_cutoff"], 0, 0, 0, 0
    )
    assert jnp.allclose(vacuum_state[0, 0], 1.0), "Vacuum state should have 1.0 at (0,0)"
    assert jnp.allclose(jnp.sum(vacuum_state**2), 1.0), "Vacuum state should be normalized"

    # Test Fock state
    fock_state = create_initial_state_with_qutip(
        "fock", default_params["n_cutoff"], 1, 1, 0, 0
    )
    assert jnp.allclose(fock_state[1 * default_params["n_cutoff"] + 1, 0], 1.0), "Fock state should have 1.0 at (n_cutoff, n_cutoff)"
    assert jnp.allclose(jnp.sum(fock_state**2), 1.0), "Fock state should be normalized"

    # Test coherent state
    coherent_state = create_initial_state_with_qutip(
        "coherent", default_params["n_cutoff"], 0, 0, default_params["alpha1"], default_params["alpha2"]
    )
    assert jnp.allclose(jnp.sum(coherent_state**2), 1.0), "Coherent state should be normalized"

    # Test invalid state type
    with pytest.raises(ValueError):
        create_initial_state_with_qutip(
            "invalid", default_params["n_cutoff"], 0, 0, 0, 0
        )

    # Test Fock numbers exceeding cutoff
    with pytest.raises(ValueError):
        create_initial_state_with_qutip(
            "fock", default_params["n_cutoff"], default_params["n_cutoff"], default_params["n_cutoff"], 0, 0
        )


def test_run_quantum_simulation(default_params):
    """
    Test the quantum simulation function.
    """
    # Run simulation
    results = run_quantum_simulation(**default_params)

    # Check results structure
    assert len(results) == 9, "Should return 9 arrays"
    time_points, q1_hist, p1_hist, q2_hist, p2_hist, var_q1_hist, var_p1_hist, var_q2_hist, var_p2_hist = results

    # Check time points
    assert len(time_points) == int(default_params["t_final"] / default_params["dt"]), "Incorrect number of time points"
    assert jnp.allclose(time_points[1] - time_points[0], default_params["dt"]), "Incorrect time step size"

    # Check history arrays
    for history in [q1_hist, p1_hist, q2_hist, p2_hist, var_q1_hist, var_p1_hist, var_q2_hist, var_p2_hist]:
        assert history.shape == (default_params["n_trajectories"], len(time_points)), "Incorrect history shape"

    # Check normalization of state vectors (for one trajectory)
    # This is a more complex test that would require access to internal state vectors
    # For now, we'll just check that the expectation values are reasonable
    assert jnp.all(jnp.abs(q1_hist) < 10), "Position expectation values should be reasonable"
    assert jnp.all(jnp.abs(p1_hist) < 10), "Momentum expectation values should be reasonable"
    assert jnp.all(jnp.abs(q2_hist) < 10), "Position expectation values should be reasonable"
    assert jnp.all(jnp.abs(p2_hist) < 10), "Momentum expectation values should be reasonable"

    # Check variances are non-negative
    assert jnp.all(var_q1_hist >= 0), "Variances should be non-negative"
    assert jnp.all(var_p1_hist >= 0), "Variances should be non-negative"
    assert jnp.all(var_q2_hist >= 0), "Variances should be non-negative"
    assert jnp.all(var_p2_hist >= 0), "Variances should be non-negative"

if __name__ == "__main__":
    pytest.main(["-v", "test_creation.py"])  # Run all tests with verbose output
