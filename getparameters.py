# -*- coding: utf-8 -*-

"""
Simulation & Pulse Parameters Module
===========================
Simple parameter class that can be imported into other files.
Change PARAMETER_SET_ID to select different parameter configurations.
Created on Mon Jun 30 16:07:57 2025 by L.K.Tarra
"""

import numpy as np

# ============================================================================
# PARAMETER SET SELECTOR
# ============================================================================
PARAMETER_SET_ID = 1  # Options: 1, 2

# ============================================================================
# PARAMETERS CLASSES
# ============================================================================

class ParametersQD:
    """Simulation parameters class."""
    
    def __init__(self, set_id=PARAMETER_SET_ID):
        """Initialize parameters based on set_id."""
        self.set_id = set_id
        # Set parameters based on ID
        if set_id == 1:
            self._set_dark_state_parameters()
        elif set_id == 2:
            self._set_alternative_parameters()
        else:
            raise ValueError(f"Unknown parameter set: {set_id}")
    
    def _set_dark_state_parameters(self):
        """Default simulation parameters."""
        self.name = "Default dark state simulation"
        
        #general parameters
        self.mu_B           = 5.7882818012e-2 #Bohr magneton (meV/T)
        self.hbar           = 0.6582173 #reduced Planck constant (meV*ps)
        self.k_B            = 8.617333e-2 #Boltzmann constant (meV/K)
        
        #QD parameters
        self.hbar_omega_X   = 1.5628e3 #exciton energy (meV)
        self.hbar_omega_D   = 1.5627e3 #dark exciton energy (meV)
        self.E_B            = 3.6 #biexciton binding energy (meV)
        self.E_XX           = 2*self.hbar_omega_X - self.E_B #biexciton energy (meV)
        self.delta_X        = 11.14e-3 #exciton energy splitting (meV)
        self.delta_D        = 11.14e-3 #dark exciton energy splitting (meV)
        self.E_X_H          = self.hbar_omega_X + 0.5*self.delta_X #horizontal exciton energy (meV)
        self.E_X_V          = self.hbar_omega_X - 0.5*self.delta_X #vertical exciton energy (meV)
        self.E_D_H          = self.hbar_omega_D + 0.5*self.delta_D #horizontal dark exciton energy (meV)
        self.E_D_V          = self.hbar_omega_D - 0.5*self.delta_D #vertical dark exciton energy (meV)
        self.g_hx           = 0.205 #hole g factor (1)
        self.g_ex           = 0.205 #electron g factor (1)
        self.Gamma_X_inv    = 180 #inverse exciton decay rate (ps)
        # dark state lifetime set to infinity (no decay)
        self.Gamma_XX_inv   = 120 #inverse biexciton decay rate (ps)
        self.QD_size        = 5 #QD size (nm)
        self.temperature    = 1.5 #QD temperature (K)
        self.B_x            = 3.4 #B field in x direction (T)
        self.B_z            = 0 #B field in z direction (T)
        
    
    def _set_alternative_parameters(self):
        """High-fidelity simulation parameters."""
        self.name = "Alternative simulation"
        
    
    def print_summary(self):
        """Print comprehensive parameter summary with organized sections."""
        print(f"\n{'='*50}")
        print(f"QUANTUM DOT SIMULATION PARAMETERS")
        print(f"{'='*50}")
        print(f"Parameter Set: {self.name}")
        print(f"Set ID: {self.set_id}")
        
        # Only print sections if parameters exist (to handle incomplete parameter sets)
        if hasattr(self, 'hbar_omega_X'):
            print(f"\n{'-'*50}")
            print("QUANTUM DOT PROPERTIES")
            print(f"{'-'*50}")
            print(f"Exciton Energy:                 {self.hbar_omega_X:>8.1f} meV")
            print(f"Dark Exciton Energy:            {self.hbar_omega_D:>8.1f} meV")
            print(f"Biexciton Binding Energy:       {self.E_B:>8.1f} meV")
            print(f"Biexciton Energy:               {self.E_XX:>8.1f} meV")
            print(f"Exciton Fine Structure:         {self.delta_X*1000:>8.2f}  ueV")
            print(f"Dark Exciton Fine Structure:    {self.delta_D*1000:>8.2f}  ueV")
            
            print(f"\nSplit Exciton Energies:")
            print(f"  Horizontal Exciton:           {self.E_X_H:>8.3f} meV")
            print(f"  Vertical Exciton:             {self.E_X_V:>8.3f} meV")
            print(f"  Horizontal Dark:              {self.E_D_H:>8.3f} meV")  
            print(f"  Vertical Dark:                {self.E_D_V:>8.3f} meV")
            
            print(f"\nCarrier Properties:")
            print(f"  Hole g-factor:                {self.g_hx:>8.3f}")
            print(f"  Electron g-factor:            {self.g_ex:>8.3f}")
            
            print(f"\nDynamics & Environment:")
            print(f"  Exciton Lifetime:             {self.Gamma_X_inv:>8.0f} ps")
            print(f"  Biexciton Lifetime:           {self.Gamma_XX_inv:>8.0f} ps")
            print(f"  QD Size:                      {self.QD_size:>8.1f} nm")
            print(f"  Temperature:                  {self.temperature:>8.1f} K")
            print(f"  Magnetic Field x:               {self.B_x:>8.1f} T")
            print(f"  Magnetic Field z:             {self.B_z:>8.1f} T")       
        
        # Derived quantities and analysis
        if hasattr(self, 'hbar_omega_X'):
            print(f"\n{'-'*50}")
            print("DERIVED QUANTITIES & ANALYSIS")
            print(f"{'-'*50}")
            
            # Zeeman splitting in magnetic field
            B_total = np.sqrt(self.B_x**2 + self.B_z**2)
            field_angle = np.arctan2(self.B_z, self.B_x) if self.B_x != 0 else np.pi/2
            zeeman_h = self.g_hx * self.mu_B * B_total  # meV/T * T = meV
            zeeman_e = self.g_ex * self.mu_B * B_total
            print(f"Zeeman Splitting (holes):      {zeeman_h:>8.2f} meV")
            print(f"Zeeman Splitting (electrons):  {zeeman_e:>8.2f} meV")
            print(f"Mixing angle of magnetic field \n (0 for pure x field):  {field_angle:>8.1f} rad")
            
            # Thermal energy comparison
            kT = self.k_B * self.temperature  # meV
            print(f"Thermal Energy (kT):           {kT:>8.3f} meV")
            print(f"kT/hbar_omega_x Ratio:         {kT/self.hbar_omega_X:>8.6f}")
            
            # Fine structure vs. thermal broadening
            fs_ratio = (self.delta_X) / kT
            print(f"Fine Structure/Thermal:        {fs_ratio:>8.2f}")

            
        print(f"\n{'='*50}")
        print("END PARAMETER SUMMARY")
        print(f"{'='*50}\n")


class DefaultPulse:
    def __init__(self, pulse_id, t_pulse_center=0):
        self.pulse_id = pulse_id
        self.t_pulse_center = t_pulse_center
        self._load_default_pulse(pulse_id)
    
    def _load_default_pulse(self, pulse_id):
        #general parameters
        self.hbar           = 0.6582173 #reduced Planck constant (meV*ps)
    
        #initialization pulse
        if pulse_id == "initialization":
            self.hbar_omega_P = 1.5610e3 #center frequency (meV)
            self.tau_0_P      = 2.9 #non-chirped pulse width (ps)
            self.GDD_P        = 0 #group delay dispersion (ps^2)
            self.Theta_P      = 4.5*np.pi #pulse area (1)
        #storage pulse
        elif pulse_id == "storage":
            self.hbar_omega_P = 1.5590e3 #center frequency (meV)
            self.tau_0_P      = 2.9 #non-chirped pulse width (ps)
            self.GDD_P        = -45 #group delay dispersion (ps^2)
            self.Theta_P      = 3.5*np.pi #pulse area (1)
        #retrieval pulse
        elif pulse_id == "retrieval":
            self.hbar_omega_P = 1.5590e3 #center frequency (meV)
            self.tau_0_P      = 2.9 #non-chirped pulse width (ps)
            self.GDD_P        = 45 #group delay dispersion (ps^2)
            self.Theta_P      = 3.5*np.pi #pulse area (1)
        else:
            raise ValueError("Invalid pulse_id. Must be 'initialization', 'storage', or 'retrieval'.")

        # compute the effective pulsewidth
        self.tau = np.sqrt(self.GDD_P**2 / (self.tau_0_P**2) + self.tau_0_P**2)
        # compute the simulated pulsewidth
        self.delta_t_sim = 3*self.tau
       
    def get_chirped_pulse_function(self):
        
        # Extract pulse parameters
        omega_P = self.hbar_omega_P / self.hbar  # Convert to angular frequency (rad/ps)
        tau_0 = self.tau_0_P
        GDD = self.GDD_P
        Theta = self.Theta_P

        # compute chirped parameters
        tau = self.tau
        a = GDD / (GDD**2 + tau_0**4)
        
        # Calculate peak amplitude from pulse area
        # For a Gaussian pulse: Theta = E0 * tau / sqrt(2*pi)
        pulse_amplitude = Theta / np.sqrt(2*np.pi * tau * tau_0)
    
        # Define the chirped pulse function
        def chirped_pulse(t):
            # Gaussian envelope
            envelope = pulse_amplitude * np.exp(-(t-self.t_pulse_center)**2 / (2 * tau**2))
        
            # Chirp phase (includes GDD term)
            # Phase = omega0*t + GDD*t^2/2
            phase = (omega_P + 0.5*a*(t-self.t_pulse_center)) * (t-self.t_pulse_center)

            # Return complex electric field
            return envelope * np.exp(-1j * phase)
    
        return chirped_pulse

    def print_summary(self):
        # Pulse parameters
        if hasattr(self, "hbar_omega_P"):
            print(f"\n{'-'*50}")
            print("OPTICAL PULSE PARAMETERS")
            print(f"{'-'*50}")

            if self.pulse_id == "initialization":
                print(f"Initialization Pulse:")
            if self.pulse_id == "storage":
                print(f"Storage Pulse:")
            if self.pulse_id == "retrieval":
                print(f"Retrieval Pulse:")
            print(f"  Center Frequency:             {self.hbar_omega_P:>8.1f} meV")
            print(f"  Pulse Width:                  {self.tau_0_P:>8.1f} ps")
            print(f"  Group Delay Dispersion:       {self.GDD_P:>8.1f} ps²")
            print(f"  Pulse Area:                   {self.Theta_P/np.pi:>8.1f}*pi")
        
# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_parameters_QD(set_id=1):
    """Get parameters instance."""
    return ParametersQD(set_id)

def get_default_pulse(pulse_id, t_pulse_center):
    return DefaultPulse(pulse_id, t_pulse_center)
    
def get_available_parameter_IDs():
    """Return available parameter set IDs."""
    return [1, 2]

# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    # Test the current parameter set
    params_QD = get_parameters_QD()
    params_QD.print_summary()
    pulse_init = get_default_pulse("initialization", 0)
    pulse_storage = get_default_pulse("storage", 0)
    pulse_retrieval = get_default_pulse("retrieval", 0)
    pulse_init.print_summary()
    pulse_storage.print_summary()
    pulse_retrieval.print_summary()
    
    # # Show all available sets
    # print("Available parameter sets:")
    # for set_id in get_available_parameter_IDs():
    #     p = Parameters(set_id)
    #     print(f"Set {set_id}: {p.name}")
    
