# -*- coding: utf-8 -*-

"""
Simulation Parameters Module
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
# PARAMETERS CLASS
# ============================================================================

class Parameters:
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
        self.Gamma_XX_inv   = 120 #inverse biexciton decay rate (ps)
        self.QD_size        = 5 #QD size (nm)
        self.temperature    = 1.5 #QD temperature (K)
        self.B_x            = 3.4 #B field in x direction (T)
        self.B_z            = 0 #B field in z direction (T)
        
        #initialization pulse
        self.hbar_omega_P_i = 1.5610e3 #center frequency (meV)
        self.tau_0_P_i      = 2.9 #non-chirped pulse width (ps)
        self.GDD_P_i        = 0 #group delay dispersion (ps^2)
        self.zeta_P_i       = 0 #laser polarization angle, 0 for horizontal (rad)
        self.Theta_P_i      = 4.5*np.pi #pulse area (1)
        self.delta_t_P_i    = 0 #time delay (ps)
        #storage pulse
        self.hbar_omega_P_s = 1.5590e3 #center frequency (meV)
        self.tau_0_P_s      = 2.9 #non-chirped pulse width (ps)
        self.GDD_P_s        = -45 #group delay dispersion (ps^2)
        self.zeta_P_s       = 0 #laser polarization angle, 0 for horizontal (rad)
        self.Theta_P_s      = 3.5*np.pi #pulse area (1)
        self.delta_t_P_s    = 70 #time delay (ps)
        #retrieval pulse
        self.hbar_omega_P_r = 1.5590e3 #center frequency (meV)
        self.tau_0_P_r      = 2.9 #non-chirped pulse width (ps)
        self.GDD_P_r        = 45 #group delay dispersion (ps^2)
        self.zeta_P_r       = 0 #laser polarization angle, 0 for horizontal (rad)
        self.Theta_P_r      = 3.5*np.pi #pulse area (1)
        self.delta_t_P_r    = 1420 #time delay (ps)
        
    
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
            print(f"Exciton Energy (ℏωₓ):           {self.hbar_omega_X:>8.1f} meV")
            print(f"Dark Exciton Energy (ℏωᴅ):      {self.hbar_omega_D:>8.1f} meV")
            print(f"Biexciton Binding Energy (Eᴃ):   {self.E_B:>8.1f} meV")
            print(f"Biexciton Energy (Eₓₓ):          {self.E_XX:>8.1f} meV")
            print(f"Exciton Fine Structure (δₓ):     {self.delta_X*1000:>8.2f} μeV")
            print(f"Dark Exciton Fine Structure:     {self.delta_D*1000:>8.2f} μeV")
            
            print(f"\nSplit Exciton Energies:")
            print(f"  Horizontal Exciton (Eₓₕ):      {self.E_X_H:>8.3f} meV")
            print(f"  Vertical Exciton (Eₓᵥ):        {self.E_X_V:>8.3f} meV")
            print(f"  Horizontal Dark (Eᴅₕ):         {self.E_D_H:>8.3f} meV")  
            print(f"  Vertical Dark (Eᴅᵥ):           {self.E_D_V:>8.3f} meV")
            
            print(f"\nCarrier Properties:")
            print(f"  Hole g-factor (gₕₓ):           {self.g_hx:>8.3f}")
            print(f"  Electron g-factor (gₑₓ):       {self.g_ex:>8.3f}")
            
            print(f"\nDynamics & Environment:")
            print(f"  Exciton Lifetime (Γₓ⁻¹):       {self.Gamma_X_inv:>8.0f} ps")
            print(f"  Biexciton Lifetime (Γₓₓ⁻¹):    {self.Gamma_XX_inv:>8.0f} ps")
            print(f"  QD Size:                       {self.QD_size:>8.1f} nm")
            print(f"  Temperature:                   {self.temperature:>8.1f} K")
            print(f"  Magnetic Field (Bₓ):           {self.B_x:>8.1f} T")
            print(f"  Magnetic Field (Bz):           {self.B_z:>8.1f} T")
        
        # Pulse parameters
        pulse_params = ['hbar_omega_P_i', 'hbar_omega_P_s']
        if any(hasattr(self, param) for param in pulse_params):
            print(f"\n{'-'*50}")
            print("OPTICAL PULSE PARAMETERS")
            print(f"{'-'*50}")
            
            # Initialization pulse
            if hasattr(self, 'hbar_omega_P_i'):
                print(f"Initialization Pulse:")
                print(f"  Center Frequency (ℏωₚᵢ):      {self.hbar_omega_P_i:>8.1f} meV")
                print(f"  Pulse Width (τ₀):             {self.tau_0_P_i:>8.1f} ps")
                print(f"  Group Delay Dispersion:       {self.GDD_P_i:>8.1f} ps²")
                print(f"  Polarization Angle (ζ):       {self.zeta_P_i:>8.2f} rad")
                print(f"  Pulse Area (Θ):               {self.Theta_P_i/np.pi:>8.1f}π")
                print(f"  Time Delay (Δt):              {self.delta_t_P_i:>8.1f} ps")
            
            # Storage pulse
            if hasattr(self, 'hbar_omega_P_s'):
                print(f"\nStorage Pulse:")
                print(f"  Center Frequency (ℏωₚₛ):      {self.hbar_omega_P_s:>8.1f} meV")
                print(f"  Pulse Width (τ₀):             {self.tau_0_P_s:>8.1f} ps")
                print(f"  Group Delay Dispersion:       {self.GDD_P_s:>8.1f} ps²")
                print(f"  Polarization Angle (ζ):       {self.zeta_P_s:>8.2f} rad")
                print(f"  Pulse Area (Θ):               {self.Theta_P_s/np.pi:>8.1f}π")
                print(f"  Time Delay (Δt):              {self.delta_t_P_s:>8.1f} ps")
                
            # Retrieval pulse
            if hasattr(self, 'hbar_omega_P_r'):
                print(f"\nRetrieval Pulse:")
                print(f"  Center Frequency (ℏωₚr):      {self.hbar_omega_P_r:>8.1f} meV")
                print(f"  Pulse Width (τ₀):             {self.tau_0_P_r:>8.1f} ps")
                print(f"  Group Delay Dispersion:       {self.GDD_P_r:>8.1f} ps²")
                print(f"  Polarization Angle (ζ):       {self.zeta_P_r:>8.2f} rad")
                print(f"  Pulse Area (Θ):               {self.Theta_P_r/np.pi:>8.1f}π")
                print(f"  Time Delay (Δt):              {self.delta_t_P_r:>8.1f} ps")
        
        # Derived quantities and analysis
        if hasattr(self, 'hbar_omega_X'):
            print(f"\n{'-'*50}")
            print("DERIVED QUANTITIES & ANALYSIS")
            print(f"{'-'*50}")
            
            # Energy detunings
            if hasattr(self, 'hbar_omega_P_i'):
                detuning_i = self.hbar_omega_P_i - self.hbar_omega_X
                detuning_s = self.hbar_omega_P_s - self.hbar_omega_X
                detuning_r = self.hbar_omega_P_r - self.hbar_omega_X
                print(f"Initialization Detuning:       {detuning_i:>8.2f} meV")
                print(f"Storage Detuning:              {detuning_s:>8.2f} meV")
                print(f"Retrieval Detuning:            {detuning_r:>8.2f} meV")
            
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
            print(f"kT/ℏωₓ Ratio:                  {kT/self.hbar_omega_X:>8.6f}")
            
            # Fine structure vs. thermal broadening
            fs_ratio = (self.delta_X) / kT
            print(f"Fine Structure/Thermal:        {fs_ratio:>8.2f}")
            
            # Pulse timing analysis
            if hasattr(self, 'delta_t_P_s'):
                storage_time = self.delta_t_P_s - self.delta_t_P_i
                coherence_periods = storage_time / self.Gamma_XX_inv
                print(f"Storage Duration:              {storage_time:>8.1f} ps")
                print(f"Storage/Coherence Ratio:       {coherence_periods:>8.2f}")
        
        print(f"\n{'='*50}")
        print("END PARAMETER SUMMARY")
        print(f"{'='*50}\n")

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_parameters(set_id=1):
    """Get parameters instance."""
    return Parameters(set_id)

def get_available_parameter_IDs():
    """Return available parameter set IDs."""
    return [1, 2]

# ============================================================================
# MAIN EXECUTION (for testing)
# ============================================================================

if __name__ == "__main__":
    # Test the current parameter set
    params = get_parameters()
    params.print_summary()
    
    # # Show all available sets
    # print("Available parameter sets:")
    # for set_id in get_available_parameter_IDs():
    #     p = Parameters(set_id)
    #     print(f"Set {set_id}: {p.name}")
    