import os
import numpy as np
from math import pi, cos
from scipy.integrate import quad
from scipy.interpolate import interp1d

h   = 6.626e-27
k_B = 1.381e-16
c   = 2.998e10

def mrn(grain_size_min = 5e-7, grain_size_max = 1e-4, exponent = -3.5, rho_gr = 3, sigma_rho_gr = 1.109, wavelength = 5.47e-5):

    cross_section_constant = 3 * (exponent + 4) / (4 * rho_gr * (grain_size_max ** (exponent + 4) - grain_size_min ** (exponent + 4))) # constant before P in formual A8 and A9 of Li et al. 2024
    module_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(module_dir, 'data/dust')
    load_Q_sca_lambda = np.vstack((np.array([[0, 0]]), np.loadtxt(os.path.join(data_dir, 'scattering_efficiency_Draine_2011_fig_22.2.csv'), delimiter=',').astype('float32'), np.array([[99999, 1]])))
    load_Q_ext_lambda = np.vstack((np.array([[0, 0]]), np.loadtxt(os.path.join(data_dir, 'extinction_efficiency_Draine_2011_fig_22.3.csv'), delimiter=',').astype('float32'), np.array([[99999, 2]])))
    Q_sca_lambda = interp1d(load_Q_sca_lambda[:, 0], load_Q_sca_lambda[:, 1], kind = 'linear') # Draine 2011 page 255, Figure 22.2
    Q_ext_lambda = interp1d(load_Q_ext_lambda[:, 0], load_Q_ext_lambda[:, 1], kind = 'linear') # Draine 2011 page 255, Figure 22.3
    scattering_efficiency_integral = quad(lambda a: a ** (2 + exponent) * Q_sca_lambda(2 * pi * a / wavelength), grain_size_min, grain_size_max)[0] # integral after P in formual A8 of Li et al. 2024
    extinction_efficiency_integral = quad(lambda a: a ** (2 + exponent) * Q_ext_lambda(2 * pi * a / wavelength), grain_size_min, grain_size_max)[0] # integral after P in formual A9 of Li et al. 2024
    sca_cm_squared_per_g = cross_section_constant * scattering_efficiency_integral # total scattering cross section cm^2 per g
    ext_cm_squared_per_g = cross_section_constant * extinction_efficiency_integral

    return sca_cm_squared_per_g, ext_cm_squared_per_g

def henyey_greenstein(angle, asymmetry_constant = 0.6): # asymmetry constant taken from Draine 2011 page 242, Figure 21.4
    return 1 / (4 * pi) * (1 - asymmetry_constant * asymmetry_constant) / ((1 + asymmetry_constant * asymmetry_constant - 2 * asymmetry_constant * cos(angle)) ** (3 / 2))

def thermal_emission(wavelength = 5.47e-5, temperature = 10):

    # if h * c < 0.1 * wavelength * k_B * temperature: # Rayleigh–Jeans law
        # return 2 * c * k_B * temperature / (wavelength ** 4)
	
    return (2 * h * c ** 2 / wavelength ** 5) / (np.exp(h * c / (wavelength * k_B * temperature)) - 1)