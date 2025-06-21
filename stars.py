import numpy as np

h   = 6.626e-27
k_B = 1.381e-16
c   = 2.998e10


def blackbody(wavelength = 5.47e-5, temperature = 20000):

    return (2 * h * c ** 2 / wavelength ** 5) / (np.exp(h * c / (wavelength * k_B * temperature)) - 1)
