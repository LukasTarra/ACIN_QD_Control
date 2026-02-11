import time
import jax.numpy as jnp

class Timer:
    """
    Simple class structure for runtime checking of code.

    Use it like this:
    timer = Timer()
    timer.start()
    sim_time = timer.stop()
    """
    
    def __init__(self):
        """Initialize a new Timer instance with start_time and end_time set to None."""
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer by recording the current time using time.perf_counter()."""
        self.start_time = time.perf_counter()

    def stop(self):
        """Stop the timer by recording the current time using time.perf_counter() and print+return the elapsed time."""
        self.end_time = time.perf_counter()
        print(f"Timer stopped after {self.elapsed()} seconds.")
        
        return self.elapsed()

    def elapsed(self):
        """Calculate the elapsed time since the timer was started.

        Returns:
            float: The elapsed time in seconds.
        Raises:
            ValueError: If the timer has not been started yet.
        """
        if self.start_time is None:
            raise ValueError("Timer not started")
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time


def complex_to_real_block(M):
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
    return block_matrix.astype(jnp.float32)


def real_to_complex_block(M: jnp.ndarray) -> jnp.ndarray:
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
    
    complex_matrix = jnp.array(real_part + 1j * imag_part, dtype=jnp.complex64)

    return complex_matrix
















