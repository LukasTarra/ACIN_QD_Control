import os
import numpy as np
from matplotlib import pyplot as plt
import pyaceqd.pulsegenerator as pg
import wave_plate_control as wpc
import configparser  
import qutip as qt
from pyaceqd.four_level_system.linear import biexciton
from pyaceqd.four_level_system.linear import biexciton_dressed_states
from pyaceqd.six_level_system.linear import sixls_linear_dressed_states 
from QUtip_four_level_system import fourlevel_system 
from pyaceqd.tools import read_calibration_file
import simulator_control as simc


HBAR = 0.6582173


class generic_wave_plate():
    def __init__(self,device = None, name = 'HWP', phase = np.pi, unit = 'deg') -> None:
        self.device = device
        self.name = name
        self.phase = phase
        self.unit = unit
    
    def set_unit(self,unit):
        self.unit = unit
    
    def get_unit(self):
        return self.unit
    
    def set_phase(self,phase):
        self.phase = phase
        
    def get_phase(self):
        return self.phase
    
    def generate_matrix(self,angle = 0):
        self.wp_mat = np.exp(-1j*self.phase/2)*np.array([[np.cos(angle)**2 + np.exp(1j*self.phase)*np.sin(angle)**2, (1-np.exp(1j*self.phase))*np.cos(angle)*np.sin(angle)],[(1-np.exp(1j*self.phase))*np.cos(angle)*np.sin(angle),np.sin(angle)**2 + np.exp(1j*self.phase)*np.cos(angle)**2]])
    
    def rotate(self,pulse_object = None, angle=0, excecute=False):
        if excecute:
            self.device.set_position(angle, excecute = excecute)
            #print('To do')
            pass
        
        if pulse_object is not None:
            if self.unit == 'deg':
                # range from 0 to 360 degrees
                angle = np.mod(angle,360)
                angle = np.deg2rad(angle)
            elif self.unit == 'rad':
                angle = np.mod(angle,2*np.pi)
            
            self.generate_matrix(angle)
            if type(pulse_object) is str:
                 pulse_object = pg.load_pulse(pulse_object)
            out_pulse_object = pulse_object.copy_pulse()
            out_pulse_object.clear_all()
            
            field_add_x = np.zeros_like(pulse_object.temporal_representation_x).astype(complex)
            field_add_y = np.zeros_like(pulse_object.temporal_representation_y).astype(complex)
            for i_temp, field_x in enumerate(pulse_object.temporal_representation_x):
                field_y = pulse_object.temporal_representation_y[i_temp]
                field_add_x[i_temp] = np.matmul(self.wp_mat,np.array([field_x,field_y]))[0]
                field_add_y[i_temp] = np.matmul(self.wp_mat,np.array([field_x,field_y]))[1]

            out_pulse_object._add_time(field_add_x,field_add_y)
            return out_pulse_object
    
    def open_control(self,pulse_object = None,open_gui = True,parent_window = None,previous_control = None):
       
        control_object = wpc.waveplate_control(self,pulse_object = pulse_object, open_gui = open_gui,parent_window=parent_window,previous_control= previous_control)
        return control_object
        
    def close(self):
        pass      
        


class simulator():
    def __init__(self,qd_calibration = None,name = 'Simulator',temp_dir = '', sim_kind = 'ACE', decay = False, phonons = False, dot_size = 5,temperature = 4, hamiltonian = None, photon_pulse_bool = False, dipole_orientation = 0) -> None:
        if qd_calibration is None and hamiltonian is None:
            print('No calibration file provided')
            return None
        
                
        self.qd_calibration = qd_calibration
        self.name = name
        self.temp_dir = temp_dir
        self.decay = decay
        self.phonons = phonons
        self.dot_size = dot_size
        self.temperature = temperature
        self.sim_out = None
        #self.photon_emission = [0,0,0,0]
        #self.photon_wavelength = [0,0,0,0]
        self.photon_pulse = None
        self.photon_pulse_bool = photon_pulse_bool
        self.dipole_orientation = dipole_orientation
        if hamiltonian is not None:
            self.hamiltonian = self.read_Hamiltonian(hamiltonian)
        else:
            self.hamiltionan = None
        
        if sim_kind is None:
            # check operating system and default to 'ACE' for linux and 'Qutip' for windows
            if os.name == 'posix':
                sim_kind = 'ACE'
            elif os.name == 'nt':
                sim_kind = 'Qutip'
        self.sim_kind = sim_kind
        self.hwp = generic_wave_plate(name = 'HWP', phase = np.pi, unit = 'deg')
        self.print_info()
        
        self.decay_scale_x = 0
        self.decay_scale_y = 0
        
        self.set_ace_six_level()
        self.refresh_num_states()
        
    
    def refresh_num_states(self):
        if any(self.sim_kind.lower() == x for x in ['ace','ace_ds','qutip']):
            self.num_states = 4 
            
        if any(self.sim_kind.lower() == x for x in ['ace_6ls']):
            self.num_states = 6 
            
            
        
        self.photon_emission = []
        self.photon_wavelength = []
        self.photon_polarisation = []
        for i in range(self.num_states + self.num_states - 4): # stupid fix, but we need 8 transitions for 6 level system
            self.photon_emission.append(0)
            self.photon_wavelength.append(0)
            self.photon_polarisation.append([])
    
    def set_qd_calibration(self,qd_calibration):
        self.qd_calibration = qd_calibration
        self.print_info()
    
    def set_phonons(self,phonons):
        self.phonons = phonons
    
    def set_temperature(self,temperature):
        self.temperature = temperature    
    
    def toggle_phonons(self):
        self.phonons = not self.phonons
    
    def set_dipole_orientation(self,dipole_orientation):
        self.dipole_orientation = dipole_orientation
    
    def get_num_states(self):
        return self.num_states
    
    def set_decay(self,decay = True):
        self.decay = decay

    def print_info(self):
        print('\nSimulator connected!')
        print('Quantum dot calibration file:',self.qd_calibration)
        print('Simulation method:',self.sim_kind)
        print('Temp directory:',self.temp_dir)

    def read_Hamiltonian(self, hamiltonian_file):
        config = configparser.ConfigParser()
        config.read(hamiltonian_file) 

        dimension = int(config['GENERAL']['dimension'])
        rotatin_frame = float(config['GENERAL']['rotating_frame'])
        field_coupling = int(config['GENERAL']['field_coupling'])
        rows = []
        for i in range(dimension): 
            index = i+1 
            rows.append(config['HAMILTONIAN']['row_'+str(index)])

        
        if self.sim_kind.lower() == 'qutip':
            current_row = rows[0].split(',')
            H_energy = float(current_row[0])*qt.basis(dimension,0)*qt.basis(dimension,0).dag()
            for i in range(dimension-1):
                current_row = rows[i+1].split(',')
                H_energy += (float(current_row[i+1])-rotatin_frame)/HBAR*1e-3*qt.basis(dimension,i+1)*qt.basis(dimension,i+1).dag() 
            
            print(H_energy)
            coupling_index = []
            coupling_type = []
            for i in range(dimension):
                current_row = rows[i].split(',')
                for j in range(i+1,dimension): 
                    current_cell = str(current_row[j]).strip()
                    if current_cell[0].lower() == 'c':
                        coupling_index.append([i,j])
                        coupling_type.append(int(current_cell[1:])) 
            
            def H_qutip():
                pass
                        
                    
        return None

    def simulate(self,pulse_object,sim_dt = None, dipole_moment = 1, plot = False):
        pulse_object.set_rotating_frame(self.qd_calibration)
        self.refresh_num_states()
        if self.dipole_orientation != 0:
            pulse_object = self.hwp.rotate(pulse_object,angle=self.dipole_orientation/2,excecute=False)
        if self.sim_kind.lower() == 'ace':
            self.decay_scale_x = 0
            self.decay_scale_y = 0
            self.sim_out = self.ace_four_level(pulse_object,sim_dt,self.decay,self.phonons,plot,dipole_moment)
        
        elif self.sim_kind.lower() == 'ace_ds':
            self.decay_scale_x = 0
            self.decay_scale_y = 0
            self.sim_out = self.ace_four_level_ds(pulse_object,sim_dt,self.decay,self.phonons,plot,dipole_moment)

        elif self.sim_kind.lower() == 'qutip':
            self.decay_scale_x = 0
            self.decay_scale_y = 0
            self.sim_out =  self.qutip_four_level(pulse_object,sim_dt,self.decay,self.phonons,plot,dipole_moment*np.pi)
            
        elif self.sim_kind == 'ace_6ls':
            self.sim_out = self.ace_six_level(pulse_object,sim_dt,self.decay,plot,dipole_moment)
        
        else:
            print('No valid simulation kind provided')
        self.photon_wavelength[0] = pulse_object._Units_inverse(pulse_object.exciton_x_emission, 'nm')
        self.photon_wavelength[1] = pulse_object._Units_inverse(pulse_object.exciton_y_emission, 'nm')
        self.photon_wavelength[2] = pulse_object._Units_inverse(pulse_object.biexciton_x_emission, 'nm')
        self.photon_wavelength[3] = pulse_object._Units_inverse(pulse_object.biexciton_y_emission, 'nm')
        
        self.photon_polarisation[0] = [1,0]
        self.photon_polarisation[1] = [0,1]
        self.photon_polarisation[2] = [1,0]
        self.photon_polarisation[3] = [0,1]
            
        
        if self.decay:
            # photon emission estimated from integral over time
            self.photon_emission[0] = np.trapz(x = np.real(self.sim_out[0]),y = np.abs(self.sim_out[2]))
            # normalize to exciton lifetime
            self.photon_emission[0] *= 1/pulse_object.lifetime_exciton*(1-self.decay_scale_x)
            # plus last value of dynamics + half of last biexcion value
            self.photon_emission[0] += np.abs(self.sim_out[2][-1]) + np.abs(self.sim_out[4][-1])/2*(1-self.decay_scale_x)

            self.photon_emission[1] = np.trapz(x = np.real(self.sim_out[0]),y = np.abs(self.sim_out[3]))
            self.photon_emission[1] *= 1/pulse_object.lifetime_exciton*(1-self.decay_scale_y)
            self.photon_emission[1] += np.abs(self.sim_out[3][-1]) + np.abs(self.sim_out[4][-1])/2*(1-self.decay_scale_y)

            # for biexciton emission same, but split in two 
            self.photon_emission[2] = np.trapz(x = np.real(self.sim_out[0]),y = np.abs(self.sim_out[4]))/2
            self.photon_emission[2] *= 1/pulse_object.lifetime_biexciton*(1-self.decay_scale_x)
            self.photon_emission[2] += np.abs(self.sim_out[4][-1])/2*(1-self.decay_scale_x)

            self.photon_emission[3] = np.trapz(x = np.real(self.sim_out[0]),y = np.abs(self.sim_out[4]))/2
            self.photon_emission[3] *= 1/pulse_object.lifetime_biexciton*(1-self.decay_scale_y)
            self.photon_emission[3] += np.abs(self.sim_out[4][-1])/2*(1-self.decay_scale_y)
        else:
            self.photon_emission[0] = np.abs(self.sim_out[2][-1]) + np.abs(self.sim_out[4][-1])/2*(1-self.decay_scale_x)
            self.photon_emission[1] = np.abs(self.sim_out[3][-1]) + np.abs(self.sim_out[4][-1])/2*(1-self.decay_scale_y)
            self.photon_emission[2] = np.abs(self.sim_out[4][-1])/2*(1-self.decay_scale_x)
            self.photon_emission[3] = np.abs(self.sim_out[4][-1])/2*(1-self.decay_scale_y)
            
        if self.num_states == 6:
            #print(self.six_ls_energy)
            new_photon_energy = np.zeros(8)
            new_photon_energy[0] = self.six_ls_energy[1] - self.six_ls_energy[0]
            new_photon_energy[1] = self.six_ls_energy[2] - self.six_ls_energy[0]
            new_photon_energy[2] = self.six_ls_energy[5] - self.six_ls_energy[1]
            new_photon_energy[3] = self.six_ls_energy[5] - self.six_ls_energy[2]
            new_photon_energy[4] = self.six_ls_energy[3] - self.six_ls_energy[0]
            new_photon_energy[5] = self.six_ls_energy[4] - self.six_ls_energy[0]
            new_photon_energy[6] = self.six_ls_energy[5] - self.six_ls_energy[3]
            new_photon_energy[7] = self.six_ls_energy[5] - self.six_ls_energy[4]
            
            for i in range(8):
                #print(pulse_object._Units_inverse(pulse_object._Units(new_photon_energy[i],'meV'),'nm'))
                self.photon_wavelength[i] = pulse_object._Units_inverse(pulse_object._Units(new_photon_energy[i],'meV'),'nm')
            #for i in range(6):
                #print(pulse_object._Units_inverse(pulse_object._Units(self.six_ls_energy[i],'meV'),'nm'))
            #print(self.photon_wavelength)
            # rethink the emission wavelength.. but looks good
            
            # config = configparser.ConfigParser()
            # config.read(self.qd_calibration)
            # dark_wavelength = float(config['EMISSION']['dark_wavelength'])
            # dark_splitting = pulse_object._Units_inverse(float(config['SPLITTING']['fss_dark'])*1e-3,'meV')
            # dark_splitting = pulse_object._Units(dark_splitting,'nm')
                                                  
            # self.photon_wavelength[4] =  dark_wavelength + dark_splitting
            # self.photon_wavelength[5] =  dark_wavelength - dark_splitting
            # self.photon_wavelength[6] = self.photon_wavelength[2] + self.photon_wavelength[0]-self.photon_wavelength[4]
            # self.photon_wavelength[7] = self.photon_wavelength[3] + self.photon_wavelength[1]-self.photon_wavelength[5]
           
            
            if self.decay:
                self.photon_emission[4] = np.trapz(x = np.real(self.sim_out[0]),y = np.abs(self.sim_out[5]))
                self.photon_emission[4] *= 1/pulse_object.lifetime_exciton*(self.decay_scale_x)
                self.photon_emission[4] += np.abs(self.sim_out[5][-1]) + np.abs(self.sim_out[4][-1])/2*self.decay_scale_x
                
                self.photon_emission[5] = np.trapz(x = np.real(self.sim_out[0]),y = np.abs(self.sim_out[6]))
                self.photon_emission[5] *= 1/pulse_object.lifetime_exciton*(self.decay_scale_y)
                self.photon_emission[5] += np.abs(self.sim_out[6][-1]) + np.abs(self.sim_out[4][-1])/2*self.decay_scale_y
                
                self.photon_emission[6] = np.trapz(x = np.real(self.sim_out[0]),y = np.abs(self.sim_out[4]))/2
                self.photon_emission[6] *= 1/pulse_object.lifetime_biexciton*(self.decay_scale_x)
                self.photon_emission[6] += np.abs(self.sim_out[4][-1])/2*self.decay_scale_x
                
                self.photon_emission[7] = np.trapz(x = np.real(self.sim_out[0]),y = np.abs(self.sim_out[4]))/2
                self.photon_emission[7] *= 1/pulse_object.lifetime_biexciton*(self.decay_scale_y)
                self.photon_emission[7] += np.abs(self.sim_out[4][-1])/2*self.decay_scale_y
            else:
                self.photon_emission[4] = np.abs(self.sim_out[5][-1])  + np.abs(self.sim_out[4][-1])/2*self.decay_scale_x
                self.photon_emission[5] = np.abs(self.sim_out[6][-1]) + np.abs(self.sim_out[4][-1])/2*self.decay_scale_y
                self.photon_emission[6] = np.abs(self.sim_out[4][-1])/2*self.decay_scale_x
                self.photon_emission[7] = np.abs(self.sim_out[4][-1])/2*self.decay_scale_y
            
            self.photon_polarisation[4] = [1,0]
            self.photon_polarisation[5] = [0,1]
            self.photon_polarisation[6] = [1,0]
            self.photon_polarisation[7] = [0,1]

        if self.photon_pulse_bool:
            pass
        self.photon_pulse = pulse_object.copy_pulse()
        self.photon_pulse.clear_all()
        self.photon_pulse.add_time_field(self.sim_out[0],np.sqrt(self.sim_out[2]), polarisation=[1,0],frequency=pulse_object.exciton_x_emission,power = self.photon_emission[0])
        self.photon_pulse.add_time_field(self.sim_out[0],np.sqrt(self.sim_out[3]), polarisation=[0,1],frequency=pulse_object.exciton_y_emission, power = self.photon_emission[1])
        self.photon_pulse.add_time_field(self.sim_out[0],np.sqrt(self.sim_out[4]), polarisation=[1,0],frequency=pulse_object.biexciton_x_emission, power = self.photon_emission[2])
        self.photon_pulse.add_time_field(self.sim_out[0],np.sqrt(self.sim_out[4]), polarisation=[0,1],frequency=pulse_object.biexciton_y_emission, power = self.photon_emission[3])

        
        
        
        # biexciton_x_emission = np.abs(sim_input[4][-1])/2 
        #     biexciton_y_emission = np.abs(sim_input[4][-1])/2 
        #     exciton_x_emission = np.abs(sim_input[2][-1]) + np.abs(sim_input[4][-1])/2
        #     exciton_y_emission = np.abs(sim_input[3][-1]) + np.abs(sim_input[4][-1])/2

        # if unit.lower()[0] == 'n':
        #     plot_domain = pulse_object.wavelengths
        #     position_x_h = pulse_object._Units_inverse(pulse_object.exciton_x_emission, 'nm')
        #     position_x_v = pulse_object._Units_inverse(pulse_object.exciton_y_emission, 'nm')
        #     postion_xx_h = pulse_object._Units_inverse(pulse_object.biexciton_x_emission, 'nm')
        #     postion_xx_v = pulse_object._Units_inverse(pulse_object.biexciton_y_emission, 'nm')
        
    def get_simulation_results(self):
        return self.sim_out

    def ace_four_level(self,pulse_object,sim_dt = None, decay = False, phonons = False,plot = False,dipole_moment = 1):
        if type(pulse_object) is str:
            pulse_object = pg.load_pulse(pulse_object)
        sim_pulse_object = pulse_object.copy_pulse()
        sim_pulse_object.clear_filter()
        sim_pulse_object.add_filter_rectangle(transmission=dipole_moment,cap_transmission=False)
        sim_pulse_object.apply_frequency_filter()
        
        
        if sim_dt is None:
            sim_dt = sim_pulse_object.dt

        # find next power of 2 dt*2^n thats just greater than 10 ps 
        n = np.ceil(np.log2(10/sim_dt))     
        t_mem = sim_dt*2**n
        pulse_x, pulse_y = sim_pulse_object.generate_pulsefiles(temp_dir=self.temp_dir,precision=8)
        t,g,x,y,b = biexciton(sim_pulse_object.t0,sim_pulse_object.tend,dt=sim_dt,delta_xy=0, delta_b=4, temp_dir=self.temp_dir,
                                lindblad=decay,pulse_file_x=pulse_x,pulse_file_y=pulse_y,
                                output_ops=['|0><0|_4','|1><1|_4','|2><2|_4','|3><3|_4'],phonons=self.phonons, ae=self.dot_size,
                                temperature=self.temperature,suffix='pulse',calibration_file=self.qd_calibration,t_mem = t_mem)
        
        if plot:
            sim_pulse_object.plot_pulses(domain='nm',plot_frequ_intensity=True,sim_input = [t,g,x,y,b],sim_label=['g','x','y','b'])
        
        return [np.real(t),np.abs(g),np.abs(x),np.abs(y),np.abs(b)]
    
    def set_ace_six_level(self, b_x = None, b_z = None, b_field_frame = True):
        if b_x is not None:
            self.b_x = b_x
        else: 
            self.b_x = 0
        if b_z is not None:
            self.b_z = b_z
        else: 
            self.b_z = 0
        self.b_field_frame = b_field_frame
    
    def set_mag_field(self,b_x,b_z):
        self.b_x = b_x
        self.b_z = b_z
    
    def set_b_field_frame(self,b_field_frame):
        self.b_field_frame = b_field_frame
        
    def toggle_b_field_frame(self):
        self.b_field_frame = not self.b_field_frame
    
    def ace_six_level(self,pulse_object,sim_dt = None, decay = False, plot = False,dipole_moment = 1):
        
        if type(pulse_object) is str:
            pulse_object = pg.load_pulse(pulse_object)
        sim_pulse_object = pulse_object.copy_pulse()
        sim_pulse_object.clear_filter()
        sim_pulse_object.add_filter_rectangle(transmission = dipole_moment,cap_transmission=False)
        sim_pulse_object.apply_frequency_filter()
        
        if sim_dt is None:
            sim_dt = sim_pulse_object.dt
            
        n = np.ceil(np.log2(10/sim_dt))
        t_mem = sim_dt*2**n
        pulse_x, pulse_y = sim_pulse_object.generate_pulsefiles(temp_dir=self.temp_dir,precision=8) 
        ds_t, _, ds_occ, _, rho = sixls_linear_dressed_states(sim_pulse_object.t0,sim_pulse_object.tend,dt=sim_dt,pulse_file_x=pulse_x,pulse_file_y=pulse_y,temp_dir=self.temp_dir, suffix='pulse', initial = '|0><0|_6', rf = False, calibration_file = self.qd_calibration, bx = self.b_x, bz = self.b_z, lindblad = decay,phonons = self.phonons, ae=self.dot_size, temperature=self.temperature) 
        
        return_vector = [np.real(ds_t)]
        if self.b_field_frame:
            rho, index = self.bx_field_basis_transformation(rho,self.b_x,self.qd_calibration, bz=self.b_z)
            new_index = [index[0],index[1],index[2],index[5],index[3],index[4]]
            for i in new_index: #range(self.num_states)
                cur_return = []
                for j in range(len(ds_t)):
                    cur_return.append(np.abs(rho[j][i,i])) 
                return_vector.append(cur_return)
            
        else:
            _,_ = self.bx_field_basis_transformation(rho,self.b_x,self.qd_calibration, bz=self.b_z)
            for i in [0,1,2,5,3,4]: #range(self.num_states)
                cur_return = []
                for j in range(len(ds_t)):
                    cur_return.append(np.abs(rho[j][i,i])) 
                return_vector.append(cur_return)
        
        return return_vector
        
        #ds_t, ds_e, ds_occ, ds_color, rho = sixls_linear_dressed_states(t0,t_end,dt=dt,pulse_file_x=pulse_x,pulse_file_y=pulse_y,temp_dir='sim_dump/', suffix='_ds2',
        #                       verbose=False,initial='|5><5|_6',
                            # lindblad=False,rf = True, rf_file = ph_x,bx=bx,temperature = 1.5, ae = 5, phonons = False,
                            #  calibration_file=calib_file, make_transparent=[0,2,4])
    
    def light_dressed_states(self,pulse_object):
        if type(pulse_object) is str:
            pulse_object = pg.load_pulse(pulse_object)
        sim_pulse_object = pulse_object.copy_pulse()
        E_X, E_Y, E_S, E_F, E_B, _, _, g_ex, g_hx, g_ez, g_hz = read_calibration_file(self.qd_calibration)
        mu_b = 5.7882818012e-2   # meV/T
        hbar = 0.6582173  # meV*ps
        #field_x, field_y = sim_pulse_object.generate_field_functions() # _lab_frame
        
        
        if self.num_states == 4:
            energy_mat = [[],[],[],[]]
            def H_func(field_x,field_y):
                H = np.array([[0,field_x,field_y,0],
                            [np.conj(field_x),E_X,0,field_x],
                            [np.conj(field_y),0,E_Y,field_y],
                            [0,np.conj(field_x),np.conj(field_y),E_B]],dtype=complex)
                
                
                return H
        elif self.num_states == 6:
            energy_mat = [[],[],[],[],[],[]]
            A = -0.5*mu_b*self.b_x*(g_ex+g_hx)
            B = -0.5*mu_b*self.b_x*(g_ex-g_hx) 
            
            C = -1j*0.5*mu_b*self.b_z*(g_ez-3*g_hz)
            D = 1j*0.5*mu_b*self.b_z*(g_ez+3*g_hz)
            def H_func(field_x,field_y):
                H = np.array([[0,field_x,field_y,0,0,0],
                    [np.conj(field_x),E_X,C,A,0,field_x],
                    [np.conj(field_y),-C,E_Y,0,B,field_y],
                    [0,A,0,E_S,D,0],
                    [0,0,B,-D,E_F,0],
                    [0,np.conj(field_x),np.conj(field_y),0,0,E_B]],dtype=complex)
                return H
        
        #self.system_hamiltonian = H_func(0,0)
        
        for i, t in enumerate(sim_pulse_object.time):
            eigenvalue, eigenvector = np.linalg.eig(H_func(sim_pulse_object.temporal_representation_x[i],sim_pulse_object.temporal_representation_y[i]))
            index = []
            for vec in eigenvector:
                index.append(np.argmax(np.abs(vec)))
            for i in range(self.num_states):
                energy_mat[i].append(np.real(eigenvalue[index[i]]))
                
        return [sim_pulse_object.time,energy_mat]
    
    # def save_system_hamiltonian_numpy(self,filename):
    #     np.save(filename,self.system_hamiltonian)
    
    # def save_system_hamiltonian_txt(self,filename):
    #     if self.sim_kind == 'ace':
    #         system_ham_save_str = np.array([])
            
    #         for i in range(self.num_states):
    #             for j in range(self.num_states):
    #                  #"{}*(|1><3|_6 + |3><1|_6 )".format(-0.5*mu_b*bx*(g_ex+g_hx))
    #                 system_ham_save_str.append(str(self.system_hamiltonian[i,j])+' ')
            
    #     elif self.sim_kind == 'qutip':
    #         pass
    #     else: 
    #         self.save_system_hamiltonian_numpy(filename)
            
        
        
        
        
    
    def bx_field_basis_transformation(self,rho,bx,calibration_file, bz = 0): 
        E_X, E_Y, E_S, E_F, E_B, _, _, g_ex, g_hx, g_ez, g_hz = read_calibration_file(calibration_file)
        mu_b = 5.7882818012e-2   # meV/T
        hbar = 0.6582173  # meV*ps
        A = -0.5*mu_b*bx*(g_ex+g_hx)
        B = -0.5*mu_b*bx*(g_ex-g_hx) 
        
        #print('+ mixing'+str(A))
        #print('- mixing'+str(B))
        
        C = -1j*0.5*mu_b*bz*(g_ez-3*g_hz)
        D = 1j*0.5*mu_b*bz*(g_ez+3*g_hz) # -
        # system_op.append("i*{}*(|1><2|_6 -|2><1|_6)".format(0.5*mu_b*bz*(g_ez-3*g_hz)))
        # system_op.append("i*{}*(|4><3|_6 - |3><4|_6 )".format(-0.5*mu_b*bz*(g_ez+3*g_hz)))
        bare_state_index = np.argsort([0,E_X,E_Y,E_S,E_F,E_B])

        H = np.array([[0,0,0,0,0,0],
                    [0,E_X,C,A,0,0],
                    [0,-C,E_Y,0,B,0],
                    [0,A,0,E_S,D,0],
                    [0,0,B,-D,E_F,0],
                    [0,0,0,0,0,E_B]])

        sub_H_x_p = np.array([[E_X,A],
                              [A,E_S]])
        sub_H_x_m = np.array([[E_Y,B],
                              [B,E_F]])
        
        _, U_x_p = np.linalg.eig(sub_H_x_p)
        _, U_x_m = np.linalg.eig(sub_H_x_m) 
        
        
        self.decay_scale_x = min(abs(U_x_p[0]))**2
        self.decay_scale_y = min(abs(U_x_m[0]))**2
        
        eigenvalue, eigenvector = np.linalg.eig(H)
        
        index = []
        for vec in eigenvector:
            index.append(np.argmax(np.abs(vec)))
        
        self.six_ls_energy = np.real(eigenvalue[index])
        #print(np.real(eigenvalue[index]))
        # energy shift -> Monday!!!
        
        #print(eigenvector.transpose())
        #dress_state_index = np.argsort(eigenvalue)
        #print([0,E_X,E_Y,E_S,E_F,E_B])
        #print(bare_state_index)
        #print(dress_state_index)
        new_rho = []
        s0 = []
        s1 = []
        s2 = []
        s3 = []
        s4 = []
        s5 = []
        for r in rho: 
            new_rho.append(eigenvector.transpose()@r@eigenvector)

            s0.append(new_rho[-1][0,0])
            s1.append(new_rho[-1][1,1])
            s2.append(new_rho[-1][2,2])
            s3.append(new_rho[-1][3,3])
            s4.append(new_rho[-1][4,4])
            s5.append(new_rho[-1][5,5])
        # print(H)
        # print('bare_states')
        # print(bare_state_index)
        # print('dressed_states')
        # print(dress_state_index)

        # print(eigenvalue)
        return np.array(new_rho), index 
    
    def ace_four_level_ds(self,pulse_object,sim_dt = None, decay = False, phonons = False,plot = False,dipole_moment = 1): 
        if type(pulse_object) is str:
            pulse_object = pg.load_pulse(pulse_object)
        sim_pulse_object = pulse_object.copy_pulse()
        sim_pulse_object.clear_filter()
        sim_pulse_object.add_filter_rectangle(transmission=dipole_moment,cap_transmission=False)
        sim_pulse_object.apply_frequency_filter()
        
        if sim_dt is None:
            sim_dt = sim_pulse_object.dt

        # find next power of 2 dt*2^n thats just greater than 10 ps 
        n = np.ceil(np.log2(10/sim_dt))     
        t_mem = sim_dt*2**n
        pulse_x, pulse_y = sim_pulse_object.generate_pulsefiles(temp_dir=self.temp_dir,precision=8)
        t,ev,ds_occ,s_colors,rho = biexciton_dressed_states(sim_pulse_object.t0,sim_pulse_object.tend,dt=sim_dt,delta_xy=0, delta_b=4, temp_dir=self.temp_dir,
                                lindblad=decay,pulse_file_x=pulse_x,pulse_file_y=pulse_y,
                                output_ops=['|0><0|_4','|1><1|_4','|2><2|_4','|3><3|_4'],phonons=self.phonons, ae=self.dot_size,
                                temperature=self.temperature,suffix='pulse',calibration_file=self.qd_calibration,t_mem = t_mem)
        t = np.real(t)
        g = rho[:,0,0]
        x = rho[:,1,1]
        y = rho[:,2,2]
        b = rho[:,3,3]
    

        s0,s1,s2,s3,e0,e1,e2,e3,new_rho = self.four_level_pulse_basis_transformation(rho,sim_pulse_object)

        # e0 = ev[:,0]
        # e1 = ev[:,1]
        # e2 = ev[:,2]
        # e3 = ev[:,3]

        hbar = 0.6582173  # meV*ps

        phot_pulse = pg.PulseGenerator(t[0],t[-1],np.abs(t[1]-t[0]),calibration_file=self.qd_calibration)
        phot_x = np.array(x,dtype=complex)*np.exp(-1j*np.ones_like(x)*phot_pulse.exciton_x_emission/hbar*2*np.pi*t)
        phot_pulse._add_time(phot_x,np.zeros_like(phot_x))

        self.phot_pulse = phot_pulse
        #phot_pulse._add_time(np.array(s0,dtype=complex)*np.exp(-1j*np.array(e0,dtype=complex)/hbar*2*np.pi*phot_pulse.time),np.zeros_like(s0))
        if plot:
            sim_pulse_object.plot_pulses(domain='nm',plot_frequ_intensity=True,sim_input = [t,g,x,y,b,s0,s1,s2,s3],sim_label=['g','x','y','b','ds0','ds1','ds2','ds3'])
            # plt.figure()
            # plt.plot(t,e0,label='s0')
            # plt.plot(t,e1,label='s1')
            # plt.plot(t,e2,label='s2')
            # plt.plot(t,e3,label='s3')
            # plt.xlabel('time / ps')
            # plt.ylabel('DS energy / meV')
            # plt.title('Dressed state energy')
            # #plt.ylim([-1,1])
            
            # plt.figure()
            # plt.plot(t,np.abs(g),label= 'g')
            # plt.plot(t,np.abs(x),label= 'x')
            # plt.plot(t,np.real(rho[:,0,1]),label='gx')
            # plt.plot(t,np.real(rho[:,1,0]), label = 'xg')

            # phot_pulse.plot_pulses(domain='nm')
            # plt.show()

        return [np.real(t),np.abs(s0),np.abs(s1),np.abs(s2),np.abs(s3)]
        

    def qutip_four_level(self,pulse_object,sim_dt = None, decay = False, phonons = False,plot = False,dipole_moment = np.pi):
        if type(pulse_object) is str:
            pulse_object = pg.load_pulse(pulse_object)

        sim_pulse_object = pulse_object.copy_pulse()
        sim_pulse_object.clear_filter()
        sim_pulse_object.add_filter_rectangle(transmission=dipole_moment,cap_transmission=False)
        sim_pulse_object.apply_frequency_filter()

        if sim_dt is None:
            sim_dt = sim_pulse_object.dt

        pulse_x, pulse_y = sim_pulse_object.generate_field_functions() 

        t,g,x,y,b, _, _, _, _, _, _ = fourlevel_system(t0=sim_pulse_object.t0,tend=sim_pulse_object.tend,dt=sim_dt,calibration_file=self.qd_calibration,pulse_x=pulse_x, pulse_y=pulse_y,timeAxis='PULSE', collapse=decay,timeAxis_smart=False) 

        if plot:
            sim_pulse_object.plot_pulses(domain='nm',plot_frequ_intensity=True,sim_input = [t,g,x,y,b],sim_label=['g','x','y','b'])
        
        return [np.real(t),np.abs(g),np.abs(x),np.abs(y),np.abs(b)]
        # t_axis, g_occ, x_occ, y_occ, b_occ, polar_gx, polar_xb, polar_gb, t_axis, pulse1, pulse2

        #tbd
    

    def four_level_pulse_basis_transformation(self,rho,pulse_object): 
        E_X, E_Y, _, _, E_B, _, _, _, _, _, _ = read_calibration_file(self.qd_calibration)
        mu_b = 5.7882818012e-2   # meV/T
        hbar = 0.6582173  # meV*ps


        bare_state_index = np.argsort([0,E_X,E_Y,E_B])

        # x_upper = np.conj(pulse_object.temporal_representation_x)
        # y_upper = np.conj(pulse_object.temporal_representation_x)
        # x_lower = pulse_object.temporal_representation_x
        # y_lower = pulse_object.temporal_representation_y

        H = np.array([[0,0,0,0],
                    [0,E_X,0,0],
                    [0,0,E_Y,0],
                    [0,0,0,E_B]],dtype=complex)
        #print(eigenvector)
        #print(eigenvector.transpose())
        #dress_state_index = np.argsort(eigenvalue)
        #print([0,E_X,E_Y,E_S,E_F,E_B])
        #print(bare_state_index)
        #print(dress_state_index)
        new_rho = []
        s0 = []
        s1 = []
        s2 = []
        s3 = []

        e0 = []
        e1 = []
        e2 = []
        e3 = []
        for i, r in enumerate(rho): 
            H[0,1] = np.conj(pulse_object.temporal_representation_x[i])
            H[0,2] = np.conj(pulse_object.temporal_representation_y[i])
            H[1,3] = np.conj(pulse_object.temporal_representation_x[i])
            H[2,3] = np.conj(pulse_object.temporal_representation_y[i])

            H[1,0] = pulse_object.temporal_representation_x[i]
            H[2,0] = pulse_object.temporal_representation_y[i]
            H[3,1] = pulse_object.temporal_representation_x[i]
            H[3,2] = pulse_object.temporal_representation_y[i]

            
            eigenvalue, eigenvector = np.linalg.eig(H)
            new_rho.append(eigenvector.transpose()@r@eigenvector)

            s0.append(new_rho[-1][0,0])
            s1.append(new_rho[-1][1,1])
            s2.append(new_rho[-1][2,2])
            s3.append(new_rho[-1][3,3])

            e0.append(np.real(eigenvalue[0]))
            e1.append(np.real(eigenvalue[1]))
            e2.append(np.real(eigenvalue[2]))
            e3.append(np.real(eigenvalue[3]))
        # print(H)
        # print('bare_states')
        # print(bare_state_index)
        # print('dressed_states')
        # print(dress_state_index)

        # print(eigenvalue)
        return s0,s1,s2,s3,e0,e1,e2,e3,np.array(new_rho)


    def simulate_folder(self,folder,decay = False, phonons = False,plot = False,dipole_moment = 1, qutip = False):
        files = os.listdir(folder)
        files = [os.path.join(folder, f) for f in files]
        files.sort(key=lambda x: os.path.getmtime(x))
        c = 0
        sim_output = []
        for pulse_file in files:
            print('Simulation progress: ',str(c+1),'/',str(len(files)))
            cur_pulse = pg.load_pulse(pulse_file)
            if qutip:
                sim_output.append(self.qutip_four_level(cur_pulse,decay,phonons,plot,dipole_moment*np.pi))
            else:
                sim_output.append(self.ace_four_level(cur_pulse,decay,phonons,plot,dipole_moment))
            c += 1
        return sim_output
        
    def spectral_plot_four_level(self,pulse_object,sim_input, model = 'static', unit = 'nm',pulse_intesity = True, plot = True):
        if type(pulse_object) is str:
            pulse_object = pg.load_pulse(pulse_object)
        if model == 'static':
            biexciton_x_emission = np.abs(sim_input[4][-1])/2 
            biexciton_y_emission = np.abs(sim_input[4][-1])/2 
            exciton_x_emission = np.abs(sim_input[2][-1]) + np.abs(sim_input[4][-1])/2
            exciton_y_emission = np.abs(sim_input[3][-1]) + np.abs(sim_input[4][-1])/2

        if unit.lower()[0] == 'n':
            plot_domain = pulse_object.wavelengths
            position_x_h = pulse_object._Units_inverse(pulse_object.exciton_x_emission, 'nm')
            position_x_v = pulse_object._Units_inverse(pulse_object.exciton_y_emission, 'nm')
            postion_xx_h = pulse_object._Units_inverse(pulse_object.biexciton_x_emission, 'nm')
            postion_xx_v = pulse_object._Units_inverse(pulse_object.biexciton_y_emission, 'nm')
        elif unit.lower()[0] == 'f':
            plot_domain = pulse_object.frequencies + pulse_object.central_frequency
            position_x_h = pulse_object.exciton_x_emission
            position_x_v = pulse_object.exciton_y_emission
            postion_xx_h = pulse_object.biexciton_x_emission
            postion_xx_v = pulse_object.biexciton_y_emission
        elif unit.lower()[0] == 'm':
            plot_domain = pulse_object.energies + pulse_object.central_energy
            position_x_h = pulse_object._Units_inverse(pulse_object.exciton_x_emission, 'meV')
            position_x_v = pulse_object._Units_inverse(pulse_object.exciton_y_emission, 'meV')
            postion_xx_h = pulse_object._Units_inverse(pulse_object.biexciton_x_emission, 'meV')
            postion_xx_v = pulse_object._Units_inverse(pulse_object.biexciton_y_emission, 'meV')

        if plot:
            if pulse_intesity: 
                plot_pulse_h = np.abs(pulse_object.frequency_representation_x)**2
                plot_pulse_v = np.abs(pulse_object.frequency_representation_y)**2
            else:
                plot_pulse_h = np.abs(pulse_object.frequency_representation_x)
                plot_pulse_v = np.abs(pulse_object.frequency_representation_y)

            plt.figure()
            fig,ax_pulse = plt.subplots()
            ax_emission=ax_pulse.twinx() 
            ax_pulse.plot(plot_domain,plot_pulse_h,'b-',label='H pol')
            ax_pulse.plot(plot_domain,plot_pulse_v,'r-',label='V pol') 
            ax_pulse.set_xlabel(unit) 
            ax_pulse.set_ylim([0,np.max([plot_pulse_h,plot_pulse_v])*1.1])
            if pulse_intesity:
                ax_pulse.set_ylabel('Pulse intensity (a.u.)')
            else:
                ax_pulse.set_ylabel('Pulse amplitude (a.u.)')
            
            ax_emission.plot(np.array([1,1])*position_x_h,np.array([0,1])*exciton_x_emission,'b-',label='H pol')
            ax_emission.plot(np.array([1,1])*position_x_v,np.array([0,1])*exciton_y_emission,'r-',label='V pol')
            ax_emission.plot(np.array([1,1])*postion_xx_h,np.array([0,1])*biexciton_x_emission,'b-')
            ax_emission.plot(np.array([1,1])*postion_xx_v,np.array([0,1])*biexciton_y_emission,'r-')
            ax_emission.set_ylabel('Emission (a.u.)') 
            ax_emission.set_ylim([0,1])
        
    def open_control(self,pulse_object = None,open_gui = True,parent_window = None,previous_control = None):
        control_object = simc.simulator_control(self,pulse_object = pulse_object, open_gui = open_gui,previous_control= previous_control)
        return control_object
        
        # def open_control(self,pulse_object = None,simulation_object = None,open_gui = True,parent_window = None,previous_control = None):
        # control_object = smc.spectrometer_control(self,pulse_object = pulse_object,simulation_object = simulation_object, open_gui = open_gui,parent_window=parent_window, previous_control = previous_control)
        # return control_object



            
        pass