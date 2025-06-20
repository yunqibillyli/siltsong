import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from numpy.linalg import norm
from matplotlib import ticker
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm, PowerNorm
from math import pi, sqrt, sin, cos, acos, atan2, exp, floor, hypot


def spherical_to_cartesian(r, theta, phi):
    sin_theta = sin(theta)
    x = r * sin_theta * cos(phi)
    y = r * sin_theta * sin(phi)
    z = r * cos(theta)
    return x, y, z

def cartesian_to_spherical(x, y, z):
    r = sqrt(x ** 2 + y ** 2 + z ** 2)
    if r == 0:
        return 0, 0, 0
    theta = acos(z / r)
    phi = atan2(y, x)
    return r, theta, phi

def vector_angle(x1, y1, z1, x2, y2, z2):
    norm1 = hypot(x1, y1, z1)
    norm2 = hypot(x2, y2, z2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot = x1 * x2 + y1 * y2 + z1 * z2
    factor = dot / (norm1 * norm2)
    factor = max(-1, min(1, factor))
    return acos(factor)

def plot_density_powernorm(density_cartesian, view_length, power = 0.2):

    view_size = view_length / 2

    @np.vectorize
    def density_map(x, y):
        return float(density_cartesian(0, y, -x))

    density_grid = np.linspace(-view_size, view_size, 1001)
    depth_grid = np.linspace(-view_size, view_size, 1001)
    density_x, density_y = np.meshgrid(density_grid, density_grid)
    density_value = density_map(density_x, density_y)
    min_nonzero = np.min(density_value[density_value != 0])

    fig, ax = plt.subplots(figsize = (10, 10))

    im = ax.imshow(density_value, origin = 'lower', extent = [-view_size, view_size, -view_size, view_size], norm = PowerNorm(power), cmap = 'afmhot', interpolation = 'bilinear', aspect = 'equal')

    ax.set_xlabel("distance (cm)", fontsize = 25)
    ax.set_ylabel("distance (cm)", fontsize = 25)

    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_powerlimits((0,0))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_fontsize(25)
    ax.yaxis.get_offset_text().set_fontsize(25)

    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label('Dust Density (g/cm$^3$)', fontsize = 25)
    cbar.ax.tick_params(labelsize = 25)
    cbar_formatter = ticker.ScalarFormatter(useMathText = True)
    cbar_formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(cbar_formatter)
    cbar.ax.yaxis.get_offset_text().set_size(25)

    ax.tick_params(axis = 'both', which = 'both', labelsize = 25)
    plt.tight_layout()

    plt.show()

def radiative_transfer(view_length, inclination_degrees, resolution, density_spherical, density_cartesian, sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count):

    dr = view_length / 2 / distance_steps
    ds = dr / distance_substeps
    dphi = pi / phi_steps
    
    dw = view_length / depth
    ds_depth = dw / depth_substeps
    grid_size = view_length / resolution
    
    inclination = pi * (inclination_degrees / 180)
    sin_inc = sin(inclination)
    cos_inc = cos(inclination)

    def cartesian_to_observer(x, y, z):
        u = x * cos_inc - z * sin_inc
        v = y
        w = x * sin_inc + z * cos_inc
        return u, v, w
    
    def observer_to_cartesian(u, v, w):
        x = u * cos_inc + w * sin_inc
        y = v
        z = -u * sin_inc + w * cos_inc
        return x, y, z
    
    def observer_to_pixels(u, v, w):
        px = int(floor((u + grid_size / 2) / grid_size) + ((resolution - 1) / 2))
        py = int(floor((v + grid_size / 2) / grid_size) + ((resolution - 1) / 2))
        d = int((view_length / 2 - w) / dw)
        return px, py, d
    
    def pixels_to_observer(px, py, d):
        u = (px - ((resolution - 1) / 2)) * grid_size
        v = (py - ((resolution - 1) / 2)) * grid_size
        w = view_length / 2 - d * dw
        return u, v, w

    def propagate_center_to_first(I, r, theta):

        for i in range(distance_substeps):
    
            k_v = (ext_cm_squared_per_g + sca_cm_squared_per_g) * density_spherical(r + i * ds, theta) # attenuation coefficient, units: cm^-1
            j_v = source_function * k_v

            dI = -I * k_v * ds + j_v * ds
            I = I + dI
    
        return I

    def compute_one_angle(i):
        row = np.ones(distance_steps + 1)
        theta = acos(1 - (i + 1) / theta_steps) # all photons between (theta - dtheta) to theta are sent to the angle theta except for theta = dtheta. 
    
        for j in range(1, distance_steps + 1):
            r = j * dr
            row[j] = propagate_center_to_first(row[j - 1], r, theta)

        return row * sin(theta) # In later code, photons sent toward uniform theta. There are denser photons sent near the axes. We penalize them here

    def compute_spherical():
        results = Parallel(n_jobs = -1)(
            delayed(compute_one_angle)(i) 
            for i in range(theta_steps)
        )
        return np.array(results)
    
    spherical_array = compute_spherical()
    cubical_array = np.zeros((resolution, (resolution + 1) // 2, depth))
    image_array = np.zeros((resolution, (resolution + 1) // 2, depth))
    
    def send_photon(i, j, phi):
    
        r = j * dr
        theta = acos(1 - (i + 1) / theta_steps)
    
        intensity = spherical_array[i, j]
    
        x, y, z = spherical_to_cartesian(r, theta, phi)
        u, v, w = cartesian_to_observer(x, y, z)
        px, py, d = observer_to_pixels(u, v, w)
    
        scattering_angle = vector_angle(u, v, w, 0, 0, 1)
    
        if px >= 0 and px < resolution and py >= 0 and py <= (resolution + 1) // 2:
            increment = intensity * (1 - exp(-dr * sca_cm_squared_per_g * density_spherical(r, theta))) # now that the photon arrived, calculate the chance to scatter between r + dr
            cubical_array[px, py, d] += increment # reserved for further scattering
            image_array[px, py, d] += increment * scattering_phase_function(scattering_angle) # peel-off amount
    
        if i == theta_steps:
            return # if theta is pi / 2, don't need to send mirror photon
    
        x, y, z = spherical_to_cartesian(r, pi - theta, phi)
        u, v, w = cartesian_to_observer(x, y, z)
        px, py, d = observer_to_pixels(u, v, w)
    
        scattering_angle = vector_angle(u, v, w, 0, 0, 1)
    
        if px >= 0 and px < resolution and py >= 0 and py <= (resolution + 1) // 2:
            increment = intensity * (1 - exp(-dr * sca_cm_squared_per_g * density_spherical(r, theta)))
            cubical_array[px, py, d] += increment
            image_array[px, py, d] += increment * scattering_phase_function(scattering_angle)

    print("Sending isotropic photons from central star(s): ")
    
    for i in tqdm(range(theta_steps)):
        for j in range(1, distance_steps + 1):
            for phi in np.arange(-pi + dphi / 2, dphi / 2, dphi): # avoid phi = 0 or pi, no complications when we mirror
                send_photon(i, j, phi)
    
    image_array[:, -1, :] *= 2
    cubical_array[:, -1, :] *= 2 # for the center row, we only calculated the top half, therefore we need to compensate
    image_array[(resolution - 1) // 2, (resolution - 1) // 2, (depth - 1) // 2] += theta_steps * distance_steps * phi_steps * 2

    def propagate_any(I, x0, y0, z0, random_x, random_y, random_z, random_steps):

        x, y, z = x0, y0, z0
        
        for i in range(random_steps):
    
            x += random_x * ds_depth
            y += random_y * ds_depth
            z += random_z * ds_depth
    
            if x ** 2 + y ** 2 + z ** 2 >= view_length ** 2:
                return 0, 0, 0, 0 # photon escapes
        
            k_v = (ext_cm_squared_per_g + sca_cm_squared_per_g) * density_cartesian(x, y, z) # attenuation coefficient, units: cm^-1
            j_v = source_function * k_v
    
            dI = -I * k_v * ds_depth + j_v * ds_depth
            I = I + dI
        
        return x, y, z, I
    
    def multiple_scattering(weight, px, py, d):
    
        u, v, w = pixels_to_observer(px, py, d)
        x0, y0, z0 = observer_to_cartesian(u, v, w)
        
        while True:
    
            random_phi = 2 * pi * np.random.rand()
            random_z = 2 * np.random.rand() - 1
            random_r = sqrt(1 - random_z ** 2)
            random_x = random_r * cos(random_phi)
            random_y = random_r * sin(random_phi)
    
            random_steps = int(np.random.randint(depth * depth_substeps) + 1)
    
            I = cubical_array[px, py, d]
            scattering_angle = vector_angle(x0, y0, z0, random_x, random_y, random_z)
            I = I * scattering_phase_function(scattering_angle)
    
            x_dest, y_dest, z_dest, I_dest = propagate_any(I, x0, y0, z0, random_x, random_y, random_z, random_steps)
    
            if I_dest == 0:
                break
    
            u_dest, v_dest, w_dest = cartesian_to_observer(x_dest, y_dest, z_dest)
            px_dest, py_dest, d_dest = observer_to_pixels(u_dest, v_dest, w_dest)
    
            if px_dest >= 0 and px_dest < resolution and py_dest >= 0 and py_dest < (resolution + 1) // 2 and d_dest >= 0 and d_dest < depth:
    
                peel_off_angle = vector_angle(random_x, random_y, random_z, 0, 0, 1)
                image_array[px_dest, py_dest, d_dest] += weight * I_dest * (1 - exp(-ds_depth * sca_cm_squared_per_g * density_cartesian(x_dest, y_dest, z_dest))) * scattering_phase_function(peel_off_angle)
            
            else:
                break

    all_positions = [
        (px, py, d)
        for px in range(resolution)
        for py in range((resolution + 1) // 2)
        for d in range(depth)
    ]
    
    ms_weight = resolution * ((resolution + 1) // 2) * depth / ms_count
    
    sampled_positions = random.sample(all_positions, ms_count)

    print("Tracing multiple scattered photons: ")
    
    for px, py, d in tqdm(sampled_positions):
        multiple_scattering(ms_weight, px, py, d)

    def propagate_los(I, px, py, d):

        u, v, w = pixels_to_observer(px, py, d)
        
        for i in range(depth_substeps):
            
            x, y, z = observer_to_cartesian(u, v, w - i * ds_depth)
        
            k_v = (ext_cm_squared_per_g + sca_cm_squared_per_g) * density_cartesian(x, y, z) # attenuation coefficient, units: cm^-1
            j_v = source_function * k_v
    
            dI = -I * k_v * ds_depth + j_v * ds_depth
            I = I + dI
        
        return I

    print("Performing peel-off: ")
    
    for px in tqdm(range(resolution)):
        for py in range((resolution + 1) // 2):
            for d in reversed(range(depth - 1)):
                image_array[px, py, d] += propagate_los(image_array[px, py, d + 1], px, py, d)

    image = np.vstack((np.transpose(image_array[:, :, 0], (1, 0)), np.transpose(image_array[:, :, 0], (1, 0))[:-1, :][::-1, :]))

    return image