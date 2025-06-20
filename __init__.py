#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from numpy.linalg import norm
from scipy.integrate import quad
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm, PowerNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from math import pi, sqrt, sin, cos, acos, atan2, tanh, exp, log, floor, hypot


# In[3]:


# default dust properties MRN
grain_size_min = 5e-7 # cm, minimum grain size
grain_size_max = 1e-4 # cm, maximum grain size
exponent = -3.5 # n(a) is proportional to a^-3.5
wavelength = 5.47e-5 # cm, F547M
rho_gr = 3 # grams / cm^3, Draine 2011 page 245, intermediate between graphite (2.24 g/cm^3) and olivine (3.8 g/cm^3)
sigma_rho_gr = 1.109

grain_sizes = np.logspace(np.log10(grain_size_min), np.log10(grain_size_max), 10000) # Preparing to Plot the Grain Size Distributions
normalization_constant = (exponent + 1) / (grain_size_max ** (exponent + 1) - grain_size_min ** (exponent + 1))
grains = np.where((grain_sizes >= grain_size_min) & (grain_sizes <= grain_size_max), normalization_constant * grain_sizes ** exponent, 0)

plt.plot(grain_sizes, grains)

plt.xscale('log')
plt.yscale('log')

plt.xlabel('grain size ($cm$)')
plt.ylabel(r'n(a) ($cm^{-4}$)')
plt.title(r'MRN grain size distribution n(a)$\propto$a$^{-3.5}$')
plt.grid(True)

plt.show()


# In[6]:


cross_section_constant = 3 * (exponent + 4) / (4 * rho_gr * (grain_size_max ** (exponent + 4) - grain_size_min ** (exponent + 4))) # constant before P in formual A8 and A9 of Li et al. 2024
load_Q_sca_lambda = np.vstack((np.array([[0, 0]]), np.asarray(np.loadtxt('scattering_efficiency.csv', delimiter = ',')).astype('float32'), np.array([[99999, 1]])))
load_Q_ext_lambda = np.vstack((np.array([[0, 0]]), np.asarray(np.loadtxt('extinction_efficiency.csv', delimiter = ',')).astype('float32'), np.array([[99999, 2]])))
Q_sca_lambda = interp1d(load_Q_sca_lambda[:, 0], load_Q_sca_lambda[:, 1], kind = 'linear') # Draine 2011 page 255, Figure 22.2
Q_ext_lambda = interp1d(load_Q_ext_lambda[:, 0], load_Q_ext_lambda[:, 1], kind = 'linear') # Draine 2011 page 255, Figure 22.3
scattering_efficiency_integral = quad(lambda a: a ** (2 + exponent) * Q_sca_lambda(2 * pi * a / wavelength), grain_size_min, grain_size_max)[0] # integral after P in formual A8 of Li et al. 2024
extinction_efficiency_integral = quad(lambda a: a ** (2 + exponent) * Q_ext_lambda(2 * pi * a / wavelength), grain_size_min, grain_size_max)[0] # integral after P in formual A9 of Li et al. 2024
sca_cm_squared_per_g = cross_section_constant * scattering_efficiency_integral # total scattering cross section cm^2 per g
ext_cm_squared_per_g = cross_section_constant * extinction_efficiency_integral

grain_sizes = np.linspace(grain_size_min, grain_size_max, 1000)

scattering_efficiency = Q_sca_lambda(2 * pi * grain_sizes / wavelength)
extinction_efficiency = Q_ext_lambda(2 * pi * grain_sizes / wavelength)

plt.plot(grain_sizes, scattering_efficiency / extinction_efficiency)

plt.xscale('log')

plt.xlabel('grain size')
plt.ylabel('albedo')
plt.grid(True)

plt.show() # albedo asymptote towards 0.6 as in Draine 2011 page 242, Figure 21.4


# In[36]:


asymmetry_constant = 0.6 # Draine 2011 page 242, Figure 21.4

def henyey_greenstein(angle):
    return 1 / (4 * pi) * (1 - asymmetry_constant * asymmetry_constant) / ((1 + asymmetry_constant * asymmetry_constant - 2 * asymmetry_constant * cos(angle)) ** (3 / 2))


# In[39]:


source_function = 0 # assume no dust emission in the optical


# In[ ]:


view_length = 8
view_size = view_length / 2
distance_steps = 200
theta_steps = 360
phi_steps = 72
distance_substeps = 10
dr = view_length / 2 / distance_steps
ds = dr / distance_substeps
dphi = pi / phi_steps

resolution = 201
depth = 201
depth_substeps = 10
dw = view_length / depth
ds_depth = dw / depth_substeps
grid_size = view_length / resolution

inclination_degrees = 66
inclination = pi * (inclination_degrees / 180)
sin_inc = sin(inclination)
cos_inc = cos(inclination)

