import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from numpy.linalg import norm
from matplotlib import ticker
from collections.abc import Callable
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

def plot_density(density_cartesian, view_length, vmin = None, vmax = None, units = 'cgs'):

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

    if vmin is None:
        vmin = np.min(density_value)
    if vmax is None:
        vmax = np.max(density_value)

    im = ax.imshow(density_value, origin = 'lower', extent = [-view_size, view_size, -view_size, view_size], vmin = vmin, vmax = vmax, cmap = 'afmhot', interpolation = 'bilinear', aspect = 'equal')

    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_powerlimits((0,0))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_fontsize(25)
    ax.yaxis.get_offset_text().set_fontsize(25)

    cbar = fig.colorbar(im, ax = ax)

    if units == 'cgs':

        ax.set_xlabel("distance (cm)", fontsize = 25)
        ax.set_ylabel("distance (cm)", fontsize = 25)
        cbar.set_label('Dust Density (g/cm$^3$)', fontsize = 25)
        
    cbar.ax.tick_params(labelsize = 25)
    cbar_formatter = ticker.ScalarFormatter(useMathText = True)
    cbar_formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(cbar_formatter)
    cbar.ax.yaxis.get_offset_text().set_size(25)

    ax.tick_params(axis = 'both', which = 'both', labelsize = 25)
    plt.tight_layout()

    plt.show()

def plot_density_powernorm(density_cartesian, view_length, power = 0.2, units = 'cgs'):

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

    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_powerlimits((0,0))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_fontsize(25)
    ax.yaxis.get_offset_text().set_fontsize(25)

    cbar = fig.colorbar(im, ax = ax)

    if units == 'cgs':

        ax.set_xlabel("distance (cm)", fontsize = 25)
        ax.set_ylabel("distance (cm)", fontsize = 25)
        cbar.set_label('Dust Density (g/cm$^3$)', fontsize = 25)
        
    cbar.ax.tick_params(labelsize = 25)
    cbar_formatter = ticker.ScalarFormatter(useMathText = True)
    cbar_formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(cbar_formatter)
    cbar.ax.yaxis.get_offset_text().set_size(25)

    ax.tick_params(axis = 'both', which = 'both', labelsize = 25)
    plt.tight_layout()

    plt.show()

def plot_density_lognorm(density_cartesian, view_length, vmin = None, vmax = None, units = 'cgs'):

    view_size = view_length / 2

    @np.vectorize
    def density_map(x, y):
        return float(density_cartesian(0, y, -x))

    density_grid = np.linspace(-view_size, view_size, 1001)
    depth_grid = np.linspace(-view_size, view_size, 1001)
    density_x, density_y = np.meshgrid(density_grid, density_grid)
    density_value = density_map(density_x, density_y)
    min_nonzero = np.min(density_value[density_value != 0])
    density_value[density_value <= 0] = min_nonzero # * 0.1

    fig, ax = plt.subplots(figsize = (10, 10))

    if vmin is None:
        vmin = np.min(density_value)
    if vmax is None:
        vmax = np.max(density_value)

    im = ax.imshow(density_value, origin = 'lower', extent = [-view_size, view_size, -view_size, view_size], norm = LogNorm(vmin = vmin, vmax = vmax), cmap = 'afmhot', interpolation = 'bilinear', aspect = 'equal')

    formatter = ticker.ScalarFormatter(useMathText = True)
    formatter.set_powerlimits((0,0))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_fontsize(25)
    ax.yaxis.get_offset_text().set_fontsize(25)

    cbar = fig.colorbar(im, ax = ax)

    if units == 'cgs':

        ax.set_xlabel("distance (cm)", fontsize = 25)
        ax.set_ylabel("distance (cm)", fontsize = 25)
        cbar.set_label('Dust Density (g/cm$^3$)', fontsize = 25)
        
    cbar.ax.tick_params(labelsize = 25)
    cbar_formatter = ticker.ScalarFormatter(useMathText = True)
    cbar_formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(cbar_formatter)
    cbar.ax.yaxis.get_offset_text().set_size(25)

    ax.tick_params(axis = 'both', which = 'both', labelsize = 25)
    plt.tight_layout()

    plt.show()

def radiative_transfer(view_length, inclination_degrees, resolution, central_source, density_spherical, density_cartesian, sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, axisymmetry = False, reflection_symmetry = False, include_central_source_self = True):

    if isinstance(density_spherical, Callable):
        density_spherical_list = [density_spherical]
    elif isinstance(density_spherical, list) and all(isinstance(d, Callable) for d in density_spherical):
        density_spherical_list = density_spherical

    if len(density_spherical_list) == 1 and axisymmetry and reflection_symmetry and include_central_source_self:

        image = radiative_transfer_v1(view_length, inclination_degrees, resolution, 
                       central_source, 
                       density_spherical, density_cartesian, 
                       sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, 
                       depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count)
        return image

    else:

        if axisymmetry and reflection_symmetry:
                
            image = radiative_transfer_axisymmetric_reflection_symmetric(view_length, inclination_degrees, resolution, 
                            central_source, 
                            density_spherical, density_cartesian, 
                            sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, 
                            depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self)
            return image

        elif axisymmetry and not reflection_symmetry:
                
            image = radiative_transfer_axisymmetric_not_reflection_symmetric(view_length, inclination_degrees, resolution, 
                            central_source, 
                            density_spherical, density_cartesian, 
                            sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, 
                            depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self)
            return image

        elif not axisymmetry and reflection_symmetry:
                
            image = radiative_transfer_not_axisymmetric_reflection_symmetric(view_length, inclination_degrees, resolution, 
                            central_source, 
                            density_spherical, density_cartesian, 
                            sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, 
                            depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self)
            return image

        else:

            image = radiative_transfer_general(view_length, inclination_degrees, resolution, 
                            central_source, 
                            density_spherical, density_cartesian, 
                            sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, 
                            depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self)
            return image

    if False: # abandoned separating components feature, not useful since gas/dust/components add to the extinction of eachother's emissions

        if isinstance(density_spherical, Callable):
            density_spherical_list = [density_spherical]
        elif isinstance(density_spherical, list) and all(isinstance(d, Callable) for d in density_spherical):
            density_spherical_list = density_spherical
        else:
            raise TypeError("density_spherical must be a callable or a list of callables")

        if isinstance(density_cartesian, Callable):
            density_cartesian_list = [density_cartesian]
        elif isinstance(density_cartesian, list) and all(isinstance(d, Callable) for d in density_cartesian):
            density_cartesian_list = density_cartesian
        else:
            raise TypeError("density_cartesian must be a callable or a list of callables")

        if isinstance(sca_cm_squared_per_g, (int, float)):
            sca_list = [sca_cm_squared_per_g]
        elif isinstance(sca_cm_squared_per_g, list) and all(isinstance(s, (int, float)) for s in sca_cm_squared_per_g):
            sca_list = sca_cm_squared_per_g
        else:
            raise TypeError("sca_cm_squared_per_g must be a number or a list of numbers")

        if isinstance(ext_cm_squared_per_g, (int, float)):
            ext_list = [ext_cm_squared_per_g]
        elif isinstance(ext_cm_squared_per_g, list) and all(isinstance(s, (int, float)) for s in ext_cm_squared_per_g):
            ext_list = ext_cm_squared_per_g
        else:
            raise TypeError("ext_cm_squared_per_g must be a number or a list of numbers")

        if isinstance(source_function, (int, float)):
            source_function_list = [source_function]
        elif isinstance(source_function, list) and all(isinstance(s, (int, float)) for s in source_function):
            source_function_list = source_function
        else:
            raise TypeError("source_function must be a number or a list of numbers")

        if isinstance(scattering_phase_function, Callable):
            scattering_phase_function_list = [scattering_phase_function]
        elif isinstance(scattering_phase_function, list) and all(isinstance(d, Callable) for d in scattering_phase_function):
            scattering_phase_function_list = scattering_phase_function
        else:
            raise TypeError("scattering_phase_function must be a callable or a list of callables")

        list_lengths = [len(density_spherical_list), len(density_cartesian_list), len(sca_list), len(ext_list), len(source_function_list), len(scattering_phase_function_list),]

        if len(set(list_lengths)) != 1:
            raise ValueError(
                f"All parameter lists must have the same length, but got lengths: "
                f"density_spherical={len(density_spherical_list)}, "
                f"density_cartesian={len(density_cartesian_list)}, "
                f"sca_cm_squared_per_g={len(sca_list)}, "
                f"ext_cm_squared_per_g={len(ext_list)}, "
                f"source_function={len(source_function_list)}, "
                f"scattering_phase_function={len(scattering_phase_function_list)}."
            )

        list_length = list_lengths[0]
        images = []

        for i in range(list_length):

            if axisymmetry and reflection_symmetry:
                    
                current_image = radiative_transfer_axisymmetric_reflection_symmetric(view_length, inclination_degrees, resolution, 
                                central_source, 
                                density_spherical, density_cartesian, 
                                sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, 
                                depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self)

            elif axisymmetry and not reflection_symmetry:
                    
                current_image = radiative_transfer_axisymmetric_not_reflection_symmetric(view_length, inclination_degrees, resolution, 
                                central_source, 
                                density_spherical, density_cartesian, 
                                sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, 
                                depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self)

            elif not axisymmetry and reflection_symmetry:
                    
                current_image = radiative_transfer_not_axisymmetric_reflection_symmetric(view_length, inclination_degrees, resolution, 
                                central_source, 
                                density_spherical, density_cartesian, 
                                sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, 
                                depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self)

            else:

                current_image = radiative_transfer_general(view_length, inclination_degrees, resolution, 
                                central_source, 
                                density_spherical, density_cartesian, 
                                sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, 
                                depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self)

            images.append(current_image)

        images = np.array(images)
        image_all = np.sum(images, axis = 0)
        images = np.insert(images, 0, image_all, axis = 0)

        return images

def radiative_transfer_axisymmetric_reflection_symmetric(view_length, inclination_degrees, resolution, central_source, density_spherical, density_cartesian, sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self):

    if isinstance(density_spherical, Callable):
        density_spherical_list = [density_spherical]
    elif isinstance(density_spherical, list) and all(isinstance(d, Callable) for d in density_spherical):
        density_spherical_list = density_spherical
    else:
        raise TypeError("density_spherical must be a callable or a list of callables")

    if isinstance(density_cartesian, Callable):
        density_cartesian_list = [density_cartesian]
    elif isinstance(density_cartesian, list) and all(isinstance(d, Callable) for d in density_cartesian):
        density_cartesian_list = density_cartesian
    else:
        raise TypeError("density_cartesian must be a callable or a list of callables")

    if isinstance(sca_cm_squared_per_g, (int, float)):
        sca_list = [sca_cm_squared_per_g]
    elif isinstance(sca_cm_squared_per_g, list) and all(isinstance(s, (int, float)) for s in sca_cm_squared_per_g):
        sca_list = sca_cm_squared_per_g
    else:
        raise TypeError("sca_cm_squared_per_g must be a number or a list of numbers")

    if isinstance(ext_cm_squared_per_g, (int, float)):
        ext_list = [ext_cm_squared_per_g]
    elif isinstance(ext_cm_squared_per_g, list) and all(isinstance(s, (int, float)) for s in ext_cm_squared_per_g):
        ext_list = ext_cm_squared_per_g
    else:
        raise TypeError("ext_cm_squared_per_g must be a number or a list of numbers")

    if isinstance(source_function, (int, float)):
        source_function_list = [source_function]
    elif isinstance(source_function, list) and all(isinstance(s, (int, float)) for s in source_function):
        source_function_list = source_function
    else:
        raise TypeError("source_function must be a number or a list of numbers")

    if isinstance(scattering_phase_function, Callable):
        scattering_phase_function_list = [scattering_phase_function]
    elif isinstance(scattering_phase_function, list) and all(isinstance(d, Callable) for d in scattering_phase_function):
        scattering_phase_function_list = scattering_phase_function
    else:
        raise TypeError("scattering_phase_function must be a callable or a list of callables")

    list_lengths = [len(density_spherical_list), len(density_cartesian_list), len(sca_list), len(ext_list), len(source_function_list), len(scattering_phase_function_list),]

    if len(set(list_lengths)) != 1:
        raise ValueError(
            f"All parameter lists must have the same length, but got lengths: "
            f"density_spherical={len(density_spherical_list)}, "
            f"density_cartesian={len(density_cartesian_list)}, "
            f"sca_cm_squared_per_g={len(sca_list)}, "
            f"ext_cm_squared_per_g={len(ext_list)}, "
            f"source_function={len(source_function_list)}, "
            f"scattering_phase_function={len(scattering_phase_function_list)}."
        )
    
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

            I_cur = I

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_spherical, source_function_cur in zip(ext_list, sca_list, density_spherical_list, source_function_list):
    
                density_cur = density_spherical(r + i * ds, theta)
                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur # scattered light is handled separately
                j_v = source_function_cur * k_v_abs

                dI = -I_cur * k_v * ds + j_v * ds
                I = I + dI
    
        return I

    def compute_one_angle(i):
        row = np.ones(distance_steps + 1) * central_source
        theta = (i + 1) * pi / 2 / theta_steps # acos(1 - (i + 1) / theta_steps) if we sample isotropically
    
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

    print("Sending photons from the central source(s). ")
    
    spherical_array = compute_spherical()
    cubical_array = np.zeros((resolution, (resolution + 1) // 2, depth))
    image_array = np.zeros((resolution, (resolution + 1) // 2, depth))
    
    def send_photon(i, j, phi):
    
        r = j * dr
        theta = (i + 1) * pi / 2 / theta_steps
    
        intensity = spherical_array[i, j]
    
        x, y, z = spherical_to_cartesian(r, theta, phi)
        u, v, w = cartesian_to_observer(x, y, z)
        px, py, d = observer_to_pixels(u, v, w)
    
        scattering_angle = vector_angle(u, v, w, 0, 0, 1)
    
        if px >= 0 and px < resolution and py >= 0 and py < (resolution + 1) // 2:

            for sca_cm_squared_per_g, density_spherical, scattering_phase_function in zip(sca_list, density_spherical_list, scattering_phase_function_list):
                
                increment = intensity * (1 - exp(-dr * sca_cm_squared_per_g * density_spherical(r, theta))) # now that the photon arrived, calculate the chance to scatter between r + dr
                cubical_array[px, py, d] += increment # reserved for further scattering
                image_array[px, py, d] += increment * scattering_phase_function(scattering_angle) # peel-off amount
    
        if i == theta_steps - 1:
            return # if theta is pi / 2, don't need to send mirror photon
    
        x, y, z = spherical_to_cartesian(r, pi - theta, phi)
        u, v, w = cartesian_to_observer(x, y, z)
        px, py, d = observer_to_pixels(u, v, w)
    
        scattering_angle = vector_angle(u, v, w, 0, 0, 1)
    
        if px >= 0 and px < resolution and py >= 0 and py < (resolution + 1) // 2:

            for sca_cm_squared_per_g, density_spherical, scattering_phase_function in zip(sca_list, density_spherical_list, scattering_phase_function_list):
                
                increment = intensity * (1 - exp(-dr * sca_cm_squared_per_g * density_spherical(r, theta)))
                cubical_array[px, py, d] += increment
                image_array[px, py, d] += increment * scattering_phase_function(scattering_angle)

    print("Tracing single scattered photons: ")
    
    for i in tqdm(range(theta_steps)):
        for j in range(1, distance_steps + 1):
            for phi in np.arange(-pi + dphi / 2, dphi / 2, dphi): # avoid phi = 0 or pi, no complications when we mirror
                send_photon(i, j, phi)
    
    image_array[:, -1, :] *= 2
    cubical_array[:, -1, :] *= 2 # for the center row, we only calculated the top half of the directions
    image_array *= (pi / (2 * theta_steps)) * dphi
    cubical_array *= (pi / (2 * theta_steps)) * dphi

    if include_central_source_self: 
        image_array[(resolution - 1) // 2, (resolution - 1) // 2, (depth - 1) // 2] += central_source

    def propagate_any(I, x0, y0, z0, random_x, random_y, random_z, random_steps):

        x, y, z = x0, y0, z0
        
        for i in range(random_steps):

            I_cur = I

            x += random_x * ds_depth
            y += random_y * ds_depth
            z += random_z * ds_depth
    
            if x ** 2 + y ** 2 + z ** 2 >= view_length ** 2:
                return 0, 0, 0, 0 # photon escapes

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_cartesian, source_function_cur in zip(ext_list, sca_list, density_cartesian_list, source_function_list):
        
                density_cur = density_cartesian(x, y, z)

                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
                j_v = source_function_cur * k_v_abs
        
                dI = -I_cur * k_v * ds_depth + j_v * ds_depth
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

            if len(scattering_phase_function_list) == 1:

                scattering_phase_function = scattering_phase_function_list[0]
                I = I * scattering_phase_function(scattering_angle)

            else: # compute a weighted average of the scattering phase functions

                total_sca_cm_squared = 0
                averaged_scattering_phase_function = 0

                for sca_cm_squared_per_g, density_cartesian, scattering_phase_function in zip(sca_list, density_cartesian_list, scattering_phase_function_list):

                    sca_cm_squared_cur = density_cartesian(x0, y0, z0) * sca_cm_squared_per_g
                    total_sca_cm_squared += sca_cm_squared_cur
                    averaged_scattering_phase_function += sca_cm_squared_cur * scattering_phase_function(scattering_angle)

                if total_sca_cm_squared > 0:
                    averaged_scattering_phase_function /= total_sca_cm_squared
                else:
                    averaged_scattering_phase_function = 0

                averaged_scattering_phase_function = averaged_scattering_phase_function / total_sca_cm_squared
                
                I = I * averaged_scattering_phase_function
    
            x_dest, y_dest, z_dest, I_dest = propagate_any(I, x0, y0, z0, random_x, random_y, random_z, random_steps)
    
            if I_dest == 0:
                break
    
            u_dest, v_dest, w_dest = cartesian_to_observer(x_dest, y_dest, z_dest)
            px_dest, py_dest, d_dest = observer_to_pixels(u_dest, v_dest, w_dest)
    
            if px_dest >= 0 and px_dest < resolution and py_dest >= 0 and py_dest < (resolution + 1) // 2 and d_dest >= 0 and d_dest < depth:
    
                peel_off_angle = vector_angle(random_x, random_y, random_z, 0, 0, 1)

                for sca_cm_squared_per_g, density_cartesian, scattering_phase_function in zip(sca_list, density_cartesian_list, scattering_phase_function_list):
                    
                    image_array[px_dest, py_dest, d_dest] += weight * I_dest * (1 - exp(-ds_depth * sca_cm_squared_per_g * density_cartesian(x_dest, y_dest, z_dest))) * scattering_phase_function(peel_off_angle)

            elif px_dest >= 0 and px_dest < resolution and py_dest >= (resolution + 1) // 2 and py_dest < resolution and d_dest >= 0 and d_dest < depth:
    
                peel_off_angle = vector_angle(random_x, random_y, random_z, 0, 0, 1)

                for sca_cm_squared_per_g, density_cartesian, scattering_phase_function in zip(sca_list, density_cartesian_list, scattering_phase_function_list):
                    
                    image_array[px_dest, int(resolution - py_dest - 1), d_dest] += weight * I_dest * (1 - exp(-ds_depth * sca_cm_squared_per_g * density_cartesian(x_dest, y_dest, z_dest))) * scattering_phase_function(peel_off_angle)
            
            else:
                break

    all_positions = [
        (px, py, d)
        for px in range(resolution)
        for py in range((resolution + 1) // 2)
        for d in range(depth)
    ]
    
    ms_weight = resolution * ((resolution + 1) // 2) * depth / ms_count

    weight = ms_weight * 2 * pi * view_length / ds_depth
    
    sampled_positions = random.sample(all_positions, int(ms_count))

    print("Tracing multiple scattered photons: ")
    
    for px, py, d in tqdm(sampled_positions):
        multiple_scattering(ms_weight, px, py, d)

    def propagate_los(I, px, py, d):

        u, v, w = pixels_to_observer(px, py, d)
        
        for i in range(depth_substeps):

            I_cur = I
            
            x, y, z = observer_to_cartesian(u, v, w - i * ds_depth)

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_cartesian, source_function_cur in zip(ext_list, sca_list, density_cartesian_list, source_function_list):
        
                density_cur = density_cartesian(x, y, z)

                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
                j_v = source_function_cur * k_v_abs
    
                dI = -I_cur * k_v * ds_depth + j_v * ds_depth
                I = I + dI
        
        return I

    print("Performing peel-off: ")
    
    for px in tqdm(range(resolution)):
        for py in range((resolution + 1) // 2):
            for d in reversed(range(depth - 1)):
                image_array[px, py, d] += propagate_los(image_array[px, py, d + 1], px, py, d)

    image = np.vstack((np.transpose(image_array[:, :, 0], (1, 0)), np.transpose(image_array[:, :, 0], (1, 0))[:-1, :][::-1, :]))

    return image

def radiative_transfer_axisymmetric_not_reflection_symmetric(view_length, inclination_degrees, resolution, central_source, density_spherical, density_cartesian, sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self):

    if isinstance(density_spherical, Callable):
        density_spherical_list = [density_spherical]
    elif isinstance(density_spherical, list) and all(isinstance(d, Callable) for d in density_spherical):
        density_spherical_list = density_spherical
    else:
        raise TypeError("density_spherical must be a callable or a list of callables")

    if isinstance(density_cartesian, Callable):
        density_cartesian_list = [density_cartesian]
    elif isinstance(density_cartesian, list) and all(isinstance(d, Callable) for d in density_cartesian):
        density_cartesian_list = density_cartesian
    else:
        raise TypeError("density_cartesian must be a callable or a list of callables")

    if isinstance(sca_cm_squared_per_g, (int, float)):
        sca_list = [sca_cm_squared_per_g]
    elif isinstance(sca_cm_squared_per_g, list) and all(isinstance(s, (int, float)) for s in sca_cm_squared_per_g):
        sca_list = sca_cm_squared_per_g
    else:
        raise TypeError("sca_cm_squared_per_g must be a number or a list of numbers")

    if isinstance(ext_cm_squared_per_g, (int, float)):
        ext_list = [ext_cm_squared_per_g]
    elif isinstance(ext_cm_squared_per_g, list) and all(isinstance(s, (int, float)) for s in ext_cm_squared_per_g):
        ext_list = ext_cm_squared_per_g
    else:
        raise TypeError("ext_cm_squared_per_g must be a number or a list of numbers")

    if isinstance(source_function, (int, float)):
        source_function_list = [source_function]
    elif isinstance(source_function, list) and all(isinstance(s, (int, float)) for s in source_function):
        source_function_list = source_function
    else:
        raise TypeError("source_function must be a number or a list of numbers")

    if isinstance(scattering_phase_function, Callable):
        scattering_phase_function_list = [scattering_phase_function]
    elif isinstance(scattering_phase_function, list) and all(isinstance(d, Callable) for d in scattering_phase_function):
        scattering_phase_function_list = scattering_phase_function
    else:
        raise TypeError("scattering_phase_function must be a callable or a list of callables")

    list_lengths = [len(density_spherical_list), len(density_cartesian_list), len(sca_list), len(ext_list), len(source_function_list), len(scattering_phase_function_list),]

    if len(set(list_lengths)) != 1:
        raise ValueError(
            f"All parameter lists must have the same length, but got lengths: "
            f"density_spherical={len(density_spherical_list)}, "
            f"density_cartesian={len(density_cartesian_list)}, "
            f"sca_cm_squared_per_g={len(sca_list)}, "
            f"ext_cm_squared_per_g={len(ext_list)}, "
            f"source_function={len(source_function_list)}, "
            f"scattering_phase_function={len(scattering_phase_function_list)}."
        )
    
    dr = view_length / 2 / distance_steps
    ds = dr / distance_substeps
    dphi = pi / phi_steps
    theta_steps *= 2
    
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

            I_cur = I

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_spherical, source_function_cur in zip(ext_list, sca_list, density_spherical_list, source_function_list):
    
                density_cur = density_spherical(r + i * ds, theta)

                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
                j_v = source_function_cur * k_v_abs

                dI = -I_cur * k_v * ds + j_v * ds
                I = I + dI
    
        return I

    def compute_one_angle(i):
        row = np.ones(distance_steps + 1) * central_source
        theta = (i + 1) * pi / theta_steps # in this version we have to take care of all theta
    
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

    print("Sending photons from the central source(s). ")
    
    spherical_array = compute_spherical()
    cubical_array = np.zeros((resolution, (resolution + 1) // 2, depth))
    image_array = np.zeros((resolution, (resolution + 1) // 2, depth))
    
    def send_photon(i, j, phi):
    
        r = j * dr
        theta = (i + 1) * pi / theta_steps
    
        intensity = spherical_array[i, j]
    
        x, y, z = spherical_to_cartesian(r, theta, phi)
        u, v, w = cartesian_to_observer(x, y, z)
        px, py, d = observer_to_pixels(u, v, w)
    
        scattering_angle = vector_angle(u, v, w, 0, 0, 1)
    
        if px >= 0 and px < resolution and py >= 0 and py < (resolution + 1) // 2:

            for sca_cm_squared_per_g, density_spherical, scattering_phase_function in zip(sca_list, density_spherical_list, scattering_phase_function_list):
                
                increment = intensity * (1 - exp(-dr * sca_cm_squared_per_g * density_spherical(r, theta))) # now that the photon arrived, calculate the chance to scatter between r + dr
                cubical_array[px, py, d] += increment # reserved for further scattering
                image_array[px, py, d] += increment * scattering_phase_function(scattering_angle) # peel-off amount

    print("Tracing single scattered photons: ")
    
    for i in tqdm(range(theta_steps)):
        for j in range(1, distance_steps + 1):
            for phi in np.arange(-pi + dphi / 2, dphi / 2, dphi): # avoid phi = 0 or pi, no complications when we mirror
                send_photon(i, j, phi)
    
    image_array[:, -1, :] *= 2
    cubical_array[:, -1, :] *= 2 # for the center row, we only calculated the top half of the directions
    image_array *= (pi / theta_steps) * dphi
    cubical_array *= (pi / theta_steps) * dphi

    if include_central_source_self: 
        image_array[(resolution - 1) // 2, (resolution - 1) // 2, (depth - 1) // 2] += central_source

    def propagate_any(I, x0, y0, z0, random_x, random_y, random_z, random_steps):

        x, y, z = x0, y0, z0
        
        for i in range(random_steps):

            I_cur = I

            x += random_x * ds_depth
            y += random_y * ds_depth
            z += random_z * ds_depth
    
            if x ** 2 + y ** 2 + z ** 2 >= view_length ** 2:
                return 0, 0, 0, 0 # photon escapes

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_cartesian, source_function_cur in zip(ext_list, sca_list, density_cartesian_list, source_function_list):
        
                density_cur = density_cartesian(x, y, z)

                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
                j_v = source_function_cur * k_v_abs
        
                dI = -I_cur * k_v * ds_depth + j_v * ds_depth
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

            if len(scattering_phase_function_list) == 1:

                scattering_phase_function = scattering_phase_function_list[0]
                I = I * scattering_phase_function(scattering_angle)

            else: # compute a weighted average of the scattering phase functions

                total_sca_cm_squared = 0
                averaged_scattering_phase_function = 0

                for sca_cm_squared_per_g, density_cartesian, scattering_phase_function in zip(sca_list, density_cartesian_list, scattering_phase_function_list):

                    sca_cm_squared_cur = density_cartesian(x0, y0, z0) * sca_cm_squared_per_g
                    total_sca_cm_squared += sca_cm_squared_cur
                    averaged_scattering_phase_function += sca_cm_squared_cur * scattering_phase_function(scattering_angle)

                if total_sca_cm_squared > 0:
                    averaged_scattering_phase_function /= total_sca_cm_squared
                else:
                    averaged_scattering_phase_function = 0

                averaged_scattering_phase_function = averaged_scattering_phase_function / total_sca_cm_squared
                
                I = I * averaged_scattering_phase_function
    
            x_dest, y_dest, z_dest, I_dest = propagate_any(I, x0, y0, z0, random_x, random_y, random_z, random_steps)
    
            if I_dest == 0:
                break
    
            u_dest, v_dest, w_dest = cartesian_to_observer(x_dest, y_dest, z_dest)
            px_dest, py_dest, d_dest = observer_to_pixels(u_dest, v_dest, w_dest)
    
            if px_dest >= 0 and px_dest < resolution and py_dest >= 0 and py_dest < (resolution + 1) // 2 and d_dest >= 0 and d_dest < depth:
    
                peel_off_angle = vector_angle(random_x, random_y, random_z, 0, 0, 1)

                for sca_cm_squared_per_g, density_cartesian, scattering_phase_function in zip(sca_list, density_cartesian_list, scattering_phase_function_list):
                    
                    image_array[px_dest, py_dest, d_dest] += weight * I_dest * (1 - exp(-ds_depth * sca_cm_squared_per_g * density_cartesian(x_dest, y_dest, z_dest))) * scattering_phase_function(peel_off_angle)

            elif px_dest >= 0 and px_dest < resolution and py_dest >= (resolution + 1) // 2 and py_dest < resolution and d_dest >= 0 and d_dest < depth:
    
                peel_off_angle = vector_angle(random_x, random_y, random_z, 0, 0, 1)

                for sca_cm_squared_per_g, density_cartesian, scattering_phase_function in zip(sca_list, density_cartesian_list, scattering_phase_function_list):
                    
                    image_array[px_dest, int(resolution - py_dest - 1), d_dest] += weight * I_dest * (1 - exp(-ds_depth * sca_cm_squared_per_g * density_cartesian(x_dest, y_dest, z_dest))) * scattering_phase_function(peel_off_angle)
            
            else:
                break

    all_positions = [
        (px, py, d)
        for px in range(resolution)
        for py in range((resolution + 1) // 2)
        for d in range(depth)
    ]
    
    ms_weight = resolution * ((resolution + 1) // 2) * depth / ms_count

    weight = ms_weight * 2 * pi * view_length / ds_depth
    
    sampled_positions = random.sample(all_positions, int(ms_count))

    print("Tracing multiple scattered photons: ")
    
    for px, py, d in tqdm(sampled_positions):
        multiple_scattering(ms_weight, px, py, d)

    def propagate_los(I, px, py, d):

        u, v, w = pixels_to_observer(px, py, d)
        
        for i in range(depth_substeps):

            I_cur = I
            
            x, y, z = observer_to_cartesian(u, v, w - i * ds_depth)

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_cartesian, source_function_cur in zip(ext_list, sca_list, density_cartesian_list, source_function_list):
        
                density_cur = density_cartesian(x, y, z)

                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
                j_v = source_function_cur * k_v_abs
    
                dI = -I_cur * k_v * ds_depth + j_v * ds_depth
                I = I + dI
        
        return I

    print("Performing peel-off: ")
    
    for px in tqdm(range(resolution)):
        for py in range((resolution + 1) // 2):
            for d in reversed(range(depth - 1)):
                image_array[px, py, d] += propagate_los(image_array[px, py, d + 1], px, py, d)

    image = np.vstack((np.transpose(image_array[:, :, 0], (1, 0)), np.transpose(image_array[:, :, 0], (1, 0))[:-1, :][::-1, :]))

    return image

def radiative_transfer_not_axisymmetric_reflection_symmetric(view_length, inclination_degrees, resolution, central_source, density_spherical, density_cartesian, sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self):

    if isinstance(density_spherical, Callable):
        density_spherical_list = [density_spherical]
    elif isinstance(density_spherical, list) and all(isinstance(d, Callable) for d in density_spherical):
        density_spherical_list = density_spherical
    else:
        raise TypeError("density_spherical must be a callable or a list of callables")

    if isinstance(density_cartesian, Callable):
        density_cartesian_list = [density_cartesian]
    elif isinstance(density_cartesian, list) and all(isinstance(d, Callable) for d in density_cartesian):
        density_cartesian_list = density_cartesian
    else:
        raise TypeError("density_cartesian must be a callable or a list of callables")

    if isinstance(sca_cm_squared_per_g, (int, float)):
        sca_list = [sca_cm_squared_per_g]
    elif isinstance(sca_cm_squared_per_g, list) and all(isinstance(s, (int, float)) for s in sca_cm_squared_per_g):
        sca_list = sca_cm_squared_per_g
    else:
        raise TypeError("sca_cm_squared_per_g must be a number or a list of numbers")

    if isinstance(ext_cm_squared_per_g, (int, float)):
        ext_list = [ext_cm_squared_per_g]
    elif isinstance(ext_cm_squared_per_g, list) and all(isinstance(s, (int, float)) for s in ext_cm_squared_per_g):
        ext_list = ext_cm_squared_per_g
    else:
        raise TypeError("ext_cm_squared_per_g must be a number or a list of numbers")

    if isinstance(source_function, (int, float)):
        source_function_list = [source_function]
    elif isinstance(source_function, list) and all(isinstance(s, (int, float)) for s in source_function):
        source_function_list = source_function
    else:
        raise TypeError("source_function must be a number or a list of numbers")

    if isinstance(scattering_phase_function, Callable):
        scattering_phase_function_list = [scattering_phase_function]
    elif isinstance(scattering_phase_function, list) and all(isinstance(d, Callable) for d in scattering_phase_function):
        scattering_phase_function_list = scattering_phase_function
    else:
        raise TypeError("scattering_phase_function must be a callable or a list of callables")

    list_lengths = [len(density_spherical_list), len(density_cartesian_list), len(sca_list), len(ext_list), len(source_function_list), len(scattering_phase_function_list),]

    if len(set(list_lengths)) != 1:
        raise ValueError(
            f"All parameter lists must have the same length, but got lengths: "
            f"density_spherical={len(density_spherical_list)}, "
            f"density_cartesian={len(density_cartesian_list)}, "
            f"sca_cm_squared_per_g={len(sca_list)}, "
            f"ext_cm_squared_per_g={len(ext_list)}, "
            f"source_function={len(source_function_list)}, "
            f"scattering_phase_function={len(scattering_phase_function_list)}."
        )

    phi_steps = phi_steps * 2
    
    dr = view_length / 2 / distance_steps
    ds = dr / distance_substeps
    dphi = 2 * pi / phi_steps
    
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

    def propagate_center_to_first(I, r, theta, phi):
        
        for i in range(distance_substeps):

            I_cur = I

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_spherical, source_function_cur in zip(ext_list, sca_list, density_spherical_list, source_function_list):
    
                density_cur = density_spherical(r + i * ds, theta, phi)

                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
                j_v = source_function_cur * k_v_abs

                dI = -I_cur * k_v * ds + j_v * ds
                I = I + dI
    
        return I

    def compute_one_angle(i, j):
        row = np.ones(distance_steps + 1) * central_source
        theta = (i + 1) * pi / 2 / theta_steps  # You can change this to acos(1 - (i + 1)/theta_steps) for isotropic sampling
        phi = j * dphi + dphi / 2
    
        for k in range(1, distance_steps + 1):
            r = k * dr
            row[k] = propagate_center_to_first(row[k - 1], r, theta, phi)
    
        return row * sin(theta)
    
    def compute_spherical():
        results = Parallel(n_jobs = -1)(
            delayed(compute_one_angle)(i, j)
            for i in range(theta_steps)
            for j in range(phi_steps)
        )
        return np.array(results).reshape(theta_steps, phi_steps, distance_steps + 1)

    print("Sending photons from the central source(s). ")
    
    spherical_array = compute_spherical()
    cubical_array = np.zeros((resolution, resolution, depth))
    image_array = np.zeros((resolution, resolution, depth))
    
    def send_photon(i, j, k):
    
        r = j * dr
        theta = (i + 1) * pi / 2 / theta_steps
        phi = k * dphi + dphi / 2
    
        intensity = spherical_array[i, k, j]
    
        x, y, z = spherical_to_cartesian(r, theta, phi)
        u, v, w = cartesian_to_observer(x, y, z)
        px, py, d = observer_to_pixels(u, v, w)
    
        scattering_angle = vector_angle(u, v, w, 0, 0, 1)
    
        if px >= 0 and px < resolution and py >= 0 and py < resolution:

            for sca_cm_squared_per_g, density_spherical, scattering_phase_function in zip(sca_list, density_spherical_list, scattering_phase_function_list):
                
                increment = intensity * (1 - exp(-dr * sca_cm_squared_per_g * density_spherical(r, theta, phi))) # now that the photon arrived, calculate the chance to scatter between r + dr
                cubical_array[px, py, d] += increment # reserved for further scattering
                image_array[px, py, d] += increment * scattering_phase_function(scattering_angle) # peel-off amount
    
        if i == theta_steps - 1:
            return # if theta is pi / 2, don't need to send mirror photon
    
        x, y, z = spherical_to_cartesian(r, pi - theta, phi)
        u, v, w = cartesian_to_observer(x, y, z)
        px, py, d = observer_to_pixels(u, v, w)
    
        scattering_angle = vector_angle(u, v, w, 0, 0, 1)
    
        if px >= 0 and px < resolution and py >= 0 and py < resolution:

            for sca_cm_squared_per_g, density_spherical, scattering_phase_function in zip(sca_list, density_spherical_list, scattering_phase_function_list):
                
                increment = intensity * (1 - exp(-dr * sca_cm_squared_per_g * density_spherical(r, theta, phi)))
                cubical_array[px, py, d] += increment
                image_array[px, py, d] += increment * scattering_phase_function(scattering_angle)

    print("Tracing single scattered photons: ")
    
    for i in tqdm(range(theta_steps)):
        for j in range(1, distance_steps + 1):
            for k in range(phi_steps):
                send_photon(i, j, k)
    
    image_array *= (pi / (2 * theta_steps)) * dphi
    cubical_array *= (pi / (2 * theta_steps)) * dphi

    if include_central_source_self: 
        image_array[(resolution - 1) // 2, (resolution - 1) // 2, (depth - 1) // 2] += central_source

    def propagate_any(I, x0, y0, z0, random_x, random_y, random_z, random_steps):

        x, y, z = x0, y0, z0
        
        for i in range(random_steps):

            I_cur = I

            x += random_x * ds_depth
            y += random_y * ds_depth
            z += random_z * ds_depth
    
            if x ** 2 + y ** 2 + z ** 2 >= view_length ** 2:
                return 0, 0, 0, 0 # photon escapes

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_cartesian, source_function_cur in zip(ext_list, sca_list, density_cartesian_list, source_function_list):
        
                density_cur = density_cartesian(x, y, z)

                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
                j_v = source_function_cur * k_v_abs
        
                dI = -I_cur * k_v * ds_depth + j_v * ds_depth
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

            if len(scattering_phase_function_list) == 1:

                scattering_phase_function = scattering_phase_function_list[0] 
                I = I * scattering_phase_function(scattering_angle)

            else: # compute a weighted average of the scattering phase functions

                total_sca_cm_squared = 0
                averaged_scattering_phase_function = 0

                for sca_cm_squared_per_g, density_cartesian, scattering_phase_function in zip(sca_list, density_cartesian_list, scattering_phase_function_list):

                    sca_cm_squared_cur = density_cartesian(x0, y0, z0) * sca_cm_squared_per_g
                    total_sca_cm_squared += sca_cm_squared_cur
                    averaged_scattering_phase_function += sca_cm_squared_cur * scattering_phase_function(scattering_angle)

                if total_sca_cm_squared > 0:
                    averaged_scattering_phase_function /= total_sca_cm_squared
                else:
                    averaged_scattering_phase_function = 0

                averaged_scattering_phase_function = averaged_scattering_phase_function / total_sca_cm_squared
                
                I = I * averaged_scattering_phase_function
    
            x_dest, y_dest, z_dest, I_dest = propagate_any(I, x0, y0, z0, random_x, random_y, random_z, random_steps)
    
            if I_dest == 0:
                break
    
            u_dest, v_dest, w_dest = cartesian_to_observer(x_dest, y_dest, z_dest)
            px_dest, py_dest, d_dest = observer_to_pixels(u_dest, v_dest, w_dest)
    
            if px_dest >= 0 and px_dest < resolution and py_dest >= 0 and py_dest < resolution and d_dest >= 0 and d_dest < depth:
    
                peel_off_angle = vector_angle(random_x, random_y, random_z, 0, 0, 1)

                for sca_cm_squared_per_g, density_cartesian, scattering_phase_function in zip(sca_list, density_cartesian_list, scattering_phase_function_list):
                    
                    image_array[px_dest, py_dest, d_dest] += weight * I_dest * (1 - exp(-ds_depth * sca_cm_squared_per_g * density_cartesian(x_dest, y_dest, z_dest))) * scattering_phase_function(peel_off_angle)
            
            else:
                break

    all_positions = [
        (px, py, d)
        for px in range(resolution)
        for py in range(resolution)
        for d in range(depth)
    ]
    
    ms_weight = resolution * resolution * depth / ms_count

    weight = ms_weight * 2 * pi * view_length / ds_depth
    
    sampled_positions = random.sample(all_positions, int(ms_count))

    print("Tracing multiple scattered photons: ")
    
    for px, py, d in tqdm(sampled_positions):
        multiple_scattering(ms_weight, px, py, d)

    def propagate_los(I, px, py, d):

        u, v, w = pixels_to_observer(px, py, d)
        
        for i in range(depth_substeps):

            I_cur = I
            
            x, y, z = observer_to_cartesian(u, v, w - i * ds_depth)

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_cartesian, source_function_cur in zip(ext_list, sca_list, density_cartesian_list, source_function_list):
        
                density_cur = density_cartesian(x, y, z)

                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
                j_v = source_function_cur * k_v_abs
    
                dI = -I_cur * k_v * ds_depth + j_v * ds_depth
                I = I + dI
        
        return I

    print("Performing peel-off: ")
    
    for px in tqdm(range(resolution)):
        for py in range(resolution):
            for d in reversed(range(depth - 1)):
                image_array[px, py, d] += propagate_los(image_array[px, py, d + 1], px, py, d)

    image = np.transpose(image_array[:, :, 0], (1, 0))

    return image

def radiative_transfer_general(view_length, inclination_degrees, resolution, central_source, density_spherical, density_cartesian, sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count, include_central_source_self):

    if isinstance(density_spherical, Callable):
        density_spherical_list = [density_spherical]
    elif isinstance(density_spherical, list) and all(isinstance(d, Callable) for d in density_spherical):
        density_spherical_list = density_spherical
    else:
        raise TypeError("density_spherical must be a callable or a list of callables")

    if isinstance(density_cartesian, Callable):
        density_cartesian_list = [density_cartesian]
    elif isinstance(density_cartesian, list) and all(isinstance(d, Callable) for d in density_cartesian):
        density_cartesian_list = density_cartesian
    else:
        raise TypeError("density_cartesian must be a callable or a list of callables")

    if isinstance(sca_cm_squared_per_g, (int, float)):
        sca_list = [sca_cm_squared_per_g]
    elif isinstance(sca_cm_squared_per_g, list) and all(isinstance(s, (int, float)) for s in sca_cm_squared_per_g):
        sca_list = sca_cm_squared_per_g
    else:
        raise TypeError("sca_cm_squared_per_g must be a number or a list of numbers")

    if isinstance(ext_cm_squared_per_g, (int, float)):
        ext_list = [ext_cm_squared_per_g]
    elif isinstance(ext_cm_squared_per_g, list) and all(isinstance(s, (int, float)) for s in ext_cm_squared_per_g):
        ext_list = ext_cm_squared_per_g
    else:
        raise TypeError("ext_cm_squared_per_g must be a number or a list of numbers")

    if isinstance(source_function, (int, float)):
        source_function_list = [source_function]
    elif isinstance(source_function, list) and all(isinstance(s, (int, float)) for s in source_function):
        source_function_list = source_function
    else:
        raise TypeError("source_function must be a number or a list of numbers")

    if isinstance(scattering_phase_function, Callable):
        scattering_phase_function_list = [scattering_phase_function]
    elif isinstance(scattering_phase_function, list) and all(isinstance(d, Callable) for d in scattering_phase_function):
        scattering_phase_function_list = scattering_phase_function
    else:
        raise TypeError("scattering_phase_function must be a callable or a list of callables")

    list_lengths = [len(density_spherical_list), len(density_cartesian_list), len(sca_list), len(ext_list), len(source_function_list), len(scattering_phase_function_list),]

    if len(set(list_lengths)) != 1:
        raise ValueError(
            f"All parameter lists must have the same length, but got lengths: "
            f"density_spherical={len(density_spherical_list)}, "
            f"density_cartesian={len(density_cartesian_list)}, "
            f"sca_cm_squared_per_g={len(sca_list)}, "
            f"ext_cm_squared_per_g={len(ext_list)}, "
            f"source_function={len(source_function_list)}, "
            f"scattering_phase_function={len(scattering_phase_function_list)}."
        )

    phi_steps = phi_steps * 2
    theta_steps = theta_steps * 2
    
    dr = view_length / 2 / distance_steps
    ds = dr / distance_substeps
    dphi = 2 * pi / phi_steps
    
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

    def propagate_center_to_first(I, r, theta, phi):
        
        for i in range(distance_substeps):

            I_cur = I

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_spherical, source_function_cur in zip(ext_list, sca_list, density_spherical_list, source_function_list):
    
                density_cur = density_spherical(r + i * ds, theta, phi)

                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
                j_v = source_function_cur * k_v_abs

                dI = -I_cur * k_v * ds + j_v * ds
                I = I + dI
    
        return I

    def compute_one_angle(i, j):
        row = np.ones(distance_steps + 1) * central_source
        theta = (i + 1) * pi / theta_steps  # You can change this to acos(1 - (i + 1)/theta_steps) for isotropic sampling
        phi = j * dphi + dphi / 2
    
        for k in range(1, distance_steps + 1):
            r = k * dr
            row[k] = propagate_center_to_first(row[k - 1], r, theta, phi)
    
        return row * sin(theta)
    
    def compute_spherical():
        results = Parallel(n_jobs = -1)(
            delayed(compute_one_angle)(i, j)
            for i in range(theta_steps)
            for j in range(phi_steps)
        )
        return np.array(results).reshape(theta_steps, phi_steps, distance_steps + 1)

    print("Sending photons from the central source(s). ")
    
    spherical_array = compute_spherical()
    cubical_array = np.zeros((resolution, resolution, depth))
    image_array = np.zeros((resolution, resolution, depth))
    
    def send_photon(i, j, k):
    
        r = j * dr
        theta = (i + 1) * pi / theta_steps
        phi = k * dphi + dphi / 2
    
        intensity = spherical_array[i, k, j]
    
        x, y, z = spherical_to_cartesian(r, theta, phi)
        u, v, w = cartesian_to_observer(x, y, z)
        px, py, d = observer_to_pixels(u, v, w)
    
        scattering_angle = vector_angle(u, v, w, 0, 0, 1)
    
        if px >= 0 and px < resolution and py >= 0 and py < resolution:

            for sca_cm_squared_per_g, density_spherical, scattering_phase_function in zip(sca_list, density_spherical_list, scattering_phase_function_list):
                
                increment = intensity * (1 - exp(-dr * sca_cm_squared_per_g * density_spherical(r, theta, phi))) # now that the photon arrived, calculate the chance to scatter between r + dr
                cubical_array[px, py, d] += increment # reserved for further scattering
                image_array[px, py, d] += increment * scattering_phase_function(scattering_angle) # peel-off amount

    print("Tracing single scattered photons: ")
    
    for i in tqdm(range(theta_steps)):
        for j in range(1, distance_steps + 1):
            for k in range(phi_steps):
                send_photon(i, j, k)
    
    image_array *= (pi / theta_steps) * dphi
    cubical_array *= (pi / theta_steps) * dphi

    if include_central_source_self: 
        image_array[(resolution - 1) // 2, (resolution - 1) // 2, (depth - 1) // 2] += central_source

    def propagate_any(I, x0, y0, z0, random_x, random_y, random_z, random_steps):

        x, y, z = x0, y0, z0
        
        for i in range(random_steps):

            I_cur = I

            x += random_x * ds_depth
            y += random_y * ds_depth
            z += random_z * ds_depth
    
            if x ** 2 + y ** 2 + z ** 2 >= view_length ** 2:
                return 0, 0, 0, 0 # photon escapes

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_cartesian, source_function_cur in zip(ext_list, sca_list, density_cartesian_list, source_function_list):
        
                density_cur = density_cartesian(x, y, z)

                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
                j_v = source_function_cur * k_v_abs
        
                dI = -I_cur * k_v * ds_depth + j_v * ds_depth
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

            if len(scattering_phase_function_list) == 1:

                scattering_phase_function = scattering_phase_function_list[0]
                I = I * scattering_phase_function(scattering_angle)

            else: # compute a weighted average of the scattering phase functions

                total_sca_cm_squared = 0
                averaged_scattering_phase_function = 0

                for sca_cm_squared_per_g, density_cartesian, scattering_phase_function in zip(sca_list, density_cartesian_list, scattering_phase_function_list):

                    sca_cm_squared_cur = density_cartesian(x0, y0, z0) * sca_cm_squared_per_g
                    total_sca_cm_squared += sca_cm_squared_cur
                    averaged_scattering_phase_function += sca_cm_squared_cur * scattering_phase_function(scattering_angle)

                if total_sca_cm_squared > 0:
                    averaged_scattering_phase_function /= total_sca_cm_squared
                else:
                    averaged_scattering_phase_function = 0

                averaged_scattering_phase_function = averaged_scattering_phase_function / total_sca_cm_squared
                
                I = I * averaged_scattering_phase_function
    
            x_dest, y_dest, z_dest, I_dest = propagate_any(I, x0, y0, z0, random_x, random_y, random_z, random_steps)
    
            if I_dest == 0:
                break
    
            u_dest, v_dest, w_dest = cartesian_to_observer(x_dest, y_dest, z_dest)
            px_dest, py_dest, d_dest = observer_to_pixels(u_dest, v_dest, w_dest)
    
            if px_dest >= 0 and px_dest < resolution and py_dest >= 0 and py_dest < resolution and d_dest >= 0 and d_dest < depth:
    
                peel_off_angle = vector_angle(random_x, random_y, random_z, 0, 0, 1)

                for sca_cm_squared_per_g, density_cartesian, scattering_phase_function in zip(sca_list, density_cartesian_list, scattering_phase_function_list):
                    
                    image_array[px_dest, py_dest, d_dest] += weight * I_dest * (1 - exp(-ds_depth * sca_cm_squared_per_g * density_cartesian(x_dest, y_dest, z_dest))) * scattering_phase_function(peel_off_angle)
            
            else:
                break

    all_positions = [
        (px, py, d)
        for px in range(resolution)
        for py in range(resolution)
        for d in range(depth)
    ]
    
    ms_weight = resolution * resolution * depth / ms_count

    weight = ms_weight * 2 * pi * view_length / ds_depth
    
    sampled_positions = random.sample(all_positions, int(ms_count))

    print("Tracing multiple scattered photons: ")
    
    for px, py, d in tqdm(sampled_positions):
        multiple_scattering(ms_weight, px, py, d)

    def propagate_los(I, px, py, d):

        u, v, w = pixels_to_observer(px, py, d)
        
        for i in range(depth_substeps):

            I_cur = I
            
            x, y, z = observer_to_cartesian(u, v, w - i * ds_depth)

            for ext_cm_squared_per_g, sca_cm_squared_per_g, density_cartesian, source_function_cur in zip(ext_list, sca_list, density_cartesian_list, source_function_list):
        
                density_cur = density_cartesian(x, y, z)

                k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
                k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
                j_v = source_function_cur * k_v_abs
    
                dI = -I_cur * k_v * ds_depth + j_v * ds_depth
                I = I + dI
        
        return I

    print("Performing peel-off: ")
    
    for px in tqdm(range(resolution)):
        for py in range(resolution):
            for d in reversed(range(depth - 1)):
                image_array[px, py, d] += propagate_los(image_array[px, py, d + 1], px, py, d)

    image = np.transpose(image_array[:, :, 0], (1, 0))

    return image

def radiative_transfer_v1(view_length, inclination_degrees, resolution, central_source, density_spherical, density_cartesian, sca_cm_squared_per_g, ext_cm_squared_per_g, source_function, scattering_phase_function, depth, depth_substeps, distance_steps, distance_substeps, theta_steps, phi_steps, ms_count):

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
    
            density_cur = density_spherical(r + i * ds, theta)

            k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
            k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
            j_v = source_function * k_v_abs

            dI = -I * k_v * ds + j_v * ds
            I = I + dI
    
        return I

    def compute_one_angle(i):
        row = np.ones(distance_steps + 1) * central_source
        theta = (i + 1) * pi / 2 / theta_steps # acos(1 - (i + 1) / theta_steps) if we sample isotropically
    
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

    print("Sending photons from the central source(s). ")
    
    spherical_array = compute_spherical()
    cubical_array = np.zeros((resolution, (resolution + 1) // 2, depth))
    image_array = np.zeros((resolution, (resolution + 1) // 2, depth))
    
    def send_photon(i, j, phi):
    
        r = j * dr
        theta = (i + 1) * pi / 2 / theta_steps
    
        intensity = spherical_array[i, j]
    
        x, y, z = spherical_to_cartesian(r, theta, phi)
        u, v, w = cartesian_to_observer(x, y, z)
        px, py, d = observer_to_pixels(u, v, w)
    
        scattering_angle = vector_angle(u, v, w, 0, 0, 1)
    
        if px >= 0 and px < resolution and py >= 0 and py < (resolution + 1) // 2:
            increment = intensity * (1 - exp(-dr * sca_cm_squared_per_g * density_spherical(r, theta))) # now that the photon arrived, calculate the chance to scatter between r + dr
            cubical_array[px, py, d] += increment # reserved for further scattering
            image_array[px, py, d] += increment * scattering_phase_function(scattering_angle) # peel-off amount
    
        if i == theta_steps - 1:
            return # if theta is pi / 2, don't need to send mirror photon
    
        x, y, z = spherical_to_cartesian(r, pi - theta, phi)
        u, v, w = cartesian_to_observer(x, y, z)
        px, py, d = observer_to_pixels(u, v, w)
    
        scattering_angle = vector_angle(u, v, w, 0, 0, 1)
    
        if px >= 0 and px < resolution and py >= 0 and py < (resolution + 1) // 2:
            increment = intensity * (1 - exp(-dr * sca_cm_squared_per_g * density_spherical(r, theta)))
            cubical_array[px, py, d] += increment
            image_array[px, py, d] += increment * scattering_phase_function(scattering_angle)

    print("Tracing single scattered photons: ")
    
    for i in tqdm(range(theta_steps)):
        for j in range(1, distance_steps + 1):
            for phi in np.arange(-pi + dphi / 2, dphi / 2, dphi): # avoid phi = 0 or pi, no complications when we mirror
                send_photon(i, j, phi)
    
    image_array[:, -1, :] *= 2
    cubical_array[:, -1, :] *= 2 # for the center row, we only calculated the top half of the directions
    image_array *= (pi / (2 * theta_steps)) * dphi
    cubical_array *= (pi / (2 * theta_steps)) * dphi
    image_array[(resolution - 1) // 2, (resolution - 1) // 2, (depth - 1) // 2] += central_source

    def propagate_any(I, x0, y0, z0, random_x, random_y, random_z, random_steps):

        x, y, z = x0, y0, z0
        
        for i in range(random_steps):
    
            x += random_x * ds_depth
            y += random_y * ds_depth
            z += random_z * ds_depth
    
            if x ** 2 + y ** 2 + z ** 2 >= view_length ** 2:
                return 0, 0, 0, 0 # photon escapes
        
            density_cur = density_cartesian(x, y, z)

            k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
            k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
            j_v = source_function * k_v_abs
    
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
            
            elif px_dest >= 0 and px_dest < resolution and py_dest >= (resolution + 1) // 2 and py_dest < resolution and d_dest >= 0 and d_dest < depth:
    
                peel_off_angle = vector_angle(random_x, random_y, random_z, 0, 0, 1)
                image_array[px_dest, int(resolution - py_dest - 1), d_dest] += weight * I_dest * (1 - exp(-ds_depth * sca_cm_squared_per_g * density_cartesian(x_dest, y_dest, z_dest))) * scattering_phase_function(peel_off_angle)
            

            else:
                break

    all_positions = [
        (px, py, d)
        for px in range(resolution)
        for py in range((resolution + 1) // 2)
        for d in range(depth)
    ]
    
    ms_weight = resolution * ((resolution + 1) // 2) * depth / ms_count

    weight = ms_weight * 2 * pi * view_length / ds_depth
    
    sampled_positions = random.sample(all_positions, int(ms_count))

    print("Tracing multiple scattered photons: ")
    
    for px, py, d in tqdm(sampled_positions):
        multiple_scattering(ms_weight, px, py, d)

    def propagate_los(I, px, py, d):

        u, v, w = pixels_to_observer(px, py, d)
        
        for i in range(depth_substeps):
            
            x, y, z = observer_to_cartesian(u, v, w - i * ds_depth)
        
            density_cur = density_cartesian(x, y, z)

            k_v = ext_cm_squared_per_g * density_cur # attenuation coefficient, units: cm^-1
            k_v_abs = (ext_cm_squared_per_g - sca_cm_squared_per_g) * density_cur
            j_v = source_function * k_v_abs
    
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