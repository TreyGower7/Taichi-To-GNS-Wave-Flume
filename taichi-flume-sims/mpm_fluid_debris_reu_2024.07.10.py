#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Added above lines for external execution of the script
import taichi as ti
from taichi import tools
import numpy as np
import platform # For getting the operating system name, taichi may already have something for this
import os
import json
import math
import imageio
import save_sim as ss
import time as T
from matplotlib import cm

ti.init(arch=ti.gpu)  # Try to run on GPU
dim = input("What Simulation Dimensionality? Select: 2D or 3D [Waiting for user input...] --> ").lower().strip()

flume_shorten_ratio = 1.0 # Halve the flume length for testing purposes
flume_thin_ratio = 1.0
buffer_cells = 3  # Number of buffer cells to add around sides of the simulation domain
grid_length = 102.4 * flume_shorten_ratio  # Max length of the simulation domain in any direction [meters]

if dim == '3d' or int(dim) == 3:
    DIMENSIONS = 3
    # Wave Flume Render 3d using grid_length as the flume length in 2D & 3D
    # https://engineering.oregonstate.edu/wave-lab/facilities/large-wave-flume
    flume_length_3d = 90 * flume_shorten_ratio # meters
    flume_height_3d = 4.6 # meters
    flume_width_3d = 3.6 * flume_thin_ratio # meters

    
    # Max_water_depth_wind_storm = 2.7 # meters
    # Define Taichi fields for flume geometry
    flume_vertices = ti.Vector.field(3, dtype=float, shape=8)
    bottom = ti.field(dtype=int, shape=6)  # Change to a 1D array
    # Only way to render flume with differing colors
    backwall = ti.field(dtype=int, shape=6)
    frontwall = ti.field(dtype=int, shape=6)
    sidewalls = ti.field(dtype=int, shape=6)

    front_back_color = (166/255, 126/255, 71/255)  # Rust color in RGB
    side_color = (197/255, 198/255, 199/255)  # Light grey color in RGB
    bottom_color = (79/255, 78/255, 78/255)
    background_color = (237/255, 235/255, 235/255)
else:
    DIMENSIONS = 2


particles_per_dx = 4
particles_per_cell = particles_per_dx ** DIMENSIONS
print("NOTE: Common initial Particles-per-Cell, (PPC), are {}, {}, {}, or {}.".format(1**DIMENSIONS, 2**DIMENSIONS, 3**DIMENSIONS, 4**DIMENSIONS))
particles_per_cell =float(input("Set the PPC, [Waiting for user input...] -->:"))
# get the inverse power of the particles per cell to get the particles per dx, rounded to the nearest integer
particles_per_dx = int(round(particles_per_cell ** (1 / DIMENSIONS)))

use_vulkan_gui = False # Needed for Windows WSL currently, and any non-vulkan systems - Turn on if you want to use the faster new GUI renderer
output_gui = True # Output to GUI window (original, not GGUI which requires vulkan for GPU render)
output_png = True # Outputs png files and makes a gif out of them
print("Output frames to GUI window{}, and PNG files{}".format(" enabled" if output_gui else " disabled", " enabled" if output_png else " disabled"))

# More bits = higher resolution, more accurate simulation, but slower and more memory usage
particle_quality_bits = 13 # Bits for particle count base unit, e.g. 13 = 2^13 = 8192 particles
grid_quality_bits = 7 # Bits for grid nodes base unit in a direction, e.g. 7 = 2^7 = 128 grid nodes in a direction
quality = 6 # Resolution multiplier that affects both particles and grid nodes by multiplying their base units w.r.t. dimensions

#Using a shorter length means that grid_ratio_y and z may need to increase to be closer to 1 since the domain becomes more like a cube as opposed to a long flume. May be better to work with the full length for now
 
 
# While we are working on getting 3D working, lets reduce the flume width (z) from 3.7 to 0.5 to avoid memory limits
# 3d sims get big very fast


n_grid_base = 2 ** grid_quality_bits # Using pow-2 grid-size for improved GPU mem usage / performance 
n_grid = n_grid_base * quality

n_grid = 1024
dx, inv_dx = float(grid_length / n_grid), float(n_grid / grid_length)

# Best to use powers of 2 for mem allocation, e.g. 0.5, 0.25, 0.125, etc. 
# Note: there are buffer grid-cells on either end of each dimension for multi-cell shape-function kernels and BCs
grid_ratio_x = 1.0000
grid_ratio_y = 0.25
grid_ratio_z = 0.125
# grid_ratio_y = (2**(math.ceil((flume_height_3d / grid_length) * n_grid) - 1).bit_length()) / n_grid
# grid_ratio_z = (2**(math.ceil((flume_width_3d / grid_length) * n_grid) - 1).bit_length()) / n_grid
print("Grid Ratios: ", grid_ratio_x, grid_ratio_y, grid_ratio_z)
grid_length_x = grid_length * ti.max(0.0, ti.min(1.0, grid_ratio_x))
grid_length_y = grid_length * ti.max(0.0, ti.min(1.0, grid_ratio_y))
grid_length_z = grid_length * ti.max(0.0, ti.min(1.0, grid_ratio_z))
print("Domain Dimensions", grid_length_x, grid_length_y, grid_length_z)
n_grid_x = int(ti.max(n_grid * ti.min(grid_ratio_x, 1), 1))
n_grid_y = int(ti.max(n_grid * ti.min(grid_ratio_y, 1), 1))
n_grid_z = int(ti.max(n_grid * ti.min(grid_ratio_z, 1), 1))
# n_grid_total = int(ti.max(n_grid_x,1) * ti.max(n_grid_y,1)) 
n_grid_total = int(ti.max(n_grid_x,1) * ti.max(n_grid_y,1) * ti.max(n_grid_z,1)) # Define this to work in 2d and 3d


print("Particles-per-Dx: ", particles_per_dx)
print("Particles-per-Cell: ", particles_per_cell)
particle_spacing_ratio = 1.0 / particles_per_dx
particle_spacing =  dx * particle_spacing_ratio
particle_volume_ratio = 1.0 / particles_per_cell
particle_volume = (dx ** DIMENSIONS) * particle_volume_ratio

n_particles_default = 81000
n_particles = n_particles_default
set_particle_count_style = "auto" # "auto" or "manual" or "optimized" or "compiled"


if (set_particle_count_style == "manual"):
    n_particles = int(input("Number of Particles to Simulate: "))
    print("Number of Particles: ", n_particles)
    
elif (set_particle_count_style == "optimized"):
    n_particles_base = 2 ** particle_quality_bits # Better ways to do this, shouldnt have to set it manually
    n_particles = n_particles_base * (quality**DIMENSIONS)
    print("Number of Particles: ", n_particles)

elif (set_particle_count_style == "compiled" or set_particle_count_style == "default"):    
    n_particles = n_particles_default



# Piston Physics
paper = "Bonus 2023"
experiment = "breaking"

if experiment == "breaking" and (paper == "Mascarenas 2022" or paper == "Bonus 2023"):
    piston_amplitude = 3.6 # 
    piston_scale_factor = 0.75 # Standard deviation scaling of the piston motion's error-function
    piston_period = piston_scale_factor*3.14159265359 # Period of the piston's motion [s]
    max_water_depth_tsunami = 1.85 # SWL from Mascardenas 2022 for breaking wave, and for Shekhar et al 2020 I believe
    wave_height_expected = 1.3 # Expected wave height for the breaking wave in meters
    wave_length_expected = 2*3.14159265359 # Expected wave length for the breaking wave in meters
    
elif experiment == "unbreaking" and (paper == "Mascarenas 2022" or paper == "Bonus 2023"):
    piston_amplitude = 3.9 # 4 meters max range on piston's stroke in the OSU LWF
    piston_scale_factor = 5.0
    piston_period = piston_scale_factor*3.14159265359 # Period of the piston's motion [s]
    max_water_depth_tsunami = 2.0 # SWL from Mascardenas 2022 for breaking wave, and for Shekhar et al 2020 I believe
    wave_height_expected = 0.2
    wave_length_expected = (piston_scale_factor*2*3.14159265359)**0.5 # Expected wave length for the tsunami wave in meters

else:
    piston_amplitude = np.pi # 4 meters max range on piston's stroke in the OSU LWF
    piston_scale_factor = 1.0
    piston_time_mean = 3.14159265359
    max_water_depth_tsunami = 2.0 # SWL maximum for tsunami irregular wave at OSU LWF [m], though the flume can accept more water it may affect the wave generation
    piston_period = piston_scale_factor*3.14159265359 # Period of the piston's motion [s] (DEPRECATED?)
    wave_height_expected = 1.0 # Expected wave height for the tsunami wave in meters
    wave_length_expected = 2*3.14159265359 # Expected wave length for the tsunami wave in meters


time_shift_piston = 1.0 # Time shift for the piston motion's error-function, experiments used ~10 seconds to ensure a smooth start to the wave
piston_motion_sample_frequency = 120.0 # Sampling frequency of the piston motion's in experimental data on DesignSafe for Mascarenas 2022
piston_time_mean = (piston_scale_factor * np.pi // 120.0) * 120.0 + time_shift_piston
piston_time_stdev = piston_scale_factor * 0.707106781187 # Std. dev. for the piston motion's error-function, SF / sqrt(2)

# piston_period = float(180.0)
piston_start_x = 1 * dx / grid_length
piston_pos = np.array([0.0, 0.0, 0.0]) + (grid_length * np.array([piston_start_x / grid_length, 0.0, 0.0])) + (dx * np.array([buffer_cells, buffer_cells, buffer_cells])) # Initial [X,Y,Z] position of the piston face
piston_travel_x = piston_amplitude / grid_length
piston_wait_time = 0.0 # don't immediately start the piston, let things settle with gravity first

buffer_shift_particles = -1.0 # How many cells to shift the particles to account for position of the actual buffer nodes 
xyz_water = np.mgrid[(piston_pos[0] + buffer_shift_particles*dx + particle_spacing/2):(flume_length_3d + buffer_cells*dx + buffer_shift_particles*dx - particle_spacing/2):particle_spacing, (buffer_cells*dx + buffer_shift_particles*dx + particle_spacing/2):(max_water_depth_tsunami + buffer_cells*dx + buffer_shift_particles*dx - particle_spacing/2):particle_spacing,  (buffer_cells*dx + buffer_shift_particles*dx + particle_spacing/2):(flume_width_3d + buffer_cells*dx + buffer_shift_particles*dx - particle_spacing/2):particle_spacing].reshape(3, -1).T
# print("XYZ Water: ", xyz_water)
print("XYZ Water Shape: ", xyz_water.shape)
n_particles_water = xyz_water.shape[0]
print("Number of Water Particles: ", n_particles_water)

debris_water_gap = dx # Gap between water and debris in, Y direction, to avoid overlap/stickiness
debris_dimensions = 2.0 * np.array([0.5, 0.05, 0.1]) # Debris dimensions in meters
debris_array = np.array([16, 1, 2]) # Number of debris in the debris-field in each direction
debris_spacing_gap = np.array([1.2, 0.4, 0.4]) # Spacing between faces of debris in the debris-field, 
debris_spacing = debris_dimensions + debris_spacing_gap # Spacing between centers of debris in the debris-field
debris_field_dimensions = debris_spacing * debris_array - debris_spacing_gap # Dimensions of the debris-field
debris_field_downstream_edge = 43.8 # Downstream edge of the debris field in meters, from Mascerenas 2022 experiments 
debris_offset = np.array([debris_field_downstream_edge - debris_field_dimensions[0], max_water_depth_tsunami + debris_water_gap, (flume_width_3d - debris_field_dimensions[2]) / 2.0]) + np.array([buffer_cells*dx + buffer_shift_particles*dx, buffer_cells*dx + buffer_shift_particles*dx, buffer_cells*dx + buffer_shift_particles*dx]) # Offset of the debris-field min corner from the origin, incl. domain buffer

# Make an array to hold all the particle positions for a piece of debris
xyz_debris = np.mgrid[(debris_offset[0] + particle_spacing/2):(debris_offset[0] + debris_dimensions[0] - particle_spacing/2):particle_spacing, (debris_offset[1] + particle_spacing/2):(debris_offset[1] + debris_dimensions[1] - particle_spacing/2):particle_spacing, (debris_offset[2] + particle_spacing/2):(debris_offset[2] + debris_dimensions[2] - particle_spacing/2):particle_spacing].reshape(3, -1).T 
n_particles_debris = xyz_debris.shape[0]


# Make an array to hold all the particle positions for the debris. It will make debris_array number of xyz_debris particle positions with the correct spacing
xyz_debris_group = np.zeros((debris_array[0] * debris_array[1] * debris_array[2] * xyz_debris.shape[0], 3))
for i in range(debris_array[0]):
    for j in range(debris_array[1]):
        for k in range(debris_array[2]):
            xyz_debris_group[(i * debris_array[1] * debris_array[2] + j * debris_array[2] + k) * xyz_debris.shape[0]:(i * debris_array[1] * debris_array[2] + j * debris_array[2] + k + 1) * xyz_debris.shape[0], :] = xyz_debris + np.array([i * debris_spacing[0], j * debris_spacing[1], k * debris_spacing[2]])
n_particles_debris_group = xyz_debris_group.shape[0]

print("Debris Dimensions: ", debris_dimensions)
print("Debris Array: ", debris_array)
print("Debris Spacing Gaps: ", debris_spacing_gap)
print("Debris Group Dimensions: ", debris_field_dimensions)
print("Debris Group Offset: ", debris_offset)
print("XYZ Debris Shape: ", xyz_debris.shape)
print("XYZ Debris Group Shape: ", xyz_debris_group.shape)

# Combine the water and debris particle positions
xyz = np.concatenate((xyz_water, xyz_debris_group), axis=0)

if set_particle_count_style == "auto" or "automatic":
    n_particles = xyz.shape[0]


print("Total Number of Particles: ", n_particles)

# xyz_numpy = np.zeros((n_particles, 3))
# xyz_numpy[:xyz.shape[0], :] = xyz
# print("XYZ Numpy: ", xyz_numpy)
# print("XYZ Numpy Shape: ", xyz_numpy.shape)



# n_particles_water = (flume_height_3d, flume_height_3d, flume_width_3d, flume_length_3d)

downsampling = True
downsampling_ratio = 1000 # Downsamples by 100x
# n_particles_water = (0.9 * 0.2 * grid_length * grid_length) * n_grid_base**2


print("Number of Particles: ", n_particles)
# T.sleep(1)
print("Downsampling: {}".format(" enabled" if downsampling else "disabled"))
# T.sleep(1)
print("Number of Downsampled Particles: ", int(n_particles / downsampling_ratio))
# T.sleep(1)
print("Number of Grid-Nodes each Direction: ", n_grid_x, n_grid_y, n_grid_z)
print("dx: ", dx)


# Material Ids
material_id_dict_gns = { "Water": 5, "Sand": 6, "Debris": 0, "Plastic": 0,
                    "Piston": 0, "Boundary": 3} # GNS Mapping Dict from Dr. Kumar

material_id_dict_mpm = { "Water": 0, "Snow": 1, "Debris": 2, "Plastic": 2, "Sand": 3, "Steel": 4, 
                    "Piston": 5, "Boundary": 6} # Taichi/our Material Mapping
    
material_id_numpy = np.zeros(n_particles)
material_id_numpy[:n_particles_water] = material_id_dict_mpm["Water"]
material_id_numpy[n_particles_water:(n_particles_water + n_particles_debris_group)] = material_id_dict_mpm["Debris"]

# Material properties
p_vol, p_rho = particle_volume, 1000.0
p_mass = p_vol * p_rho
E, nu = 2e7, 0.25  # Young's modulus and Poisson's ratio
# TODO: Define material laws for various materials
gamma_water = 7.125 #Ratio of specific heats for water 
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
fps = 20
time_delta = 1.0 / fps


# Calc timestep based on elastic moduli of materials
CFL = 0.45 # CFL stability number. Typically 0.3 - 0.5 is good
bulk_modulus = E / (3 * (1 - 2 * nu))  # Bulk modulus
max_vel = math.sqrt( max(abs(bulk_modulus), 1.0) / max(abs(p_rho), 1.0) ) # Speed of sound in the material
critical_time_step = CFL * dx / max_vel # Critical time step for stability in explicit time-integration rel. to pressure wave speed
scaled_time_step = critical_time_step * 1.0 # may need to adjust based on the domain size
print("Critical Time Step: ", critical_time_step)
print("Scaled Time Step: ", scaled_time_step)
set_dt_to_critical = True
if set_dt_to_critical:
    print("Using CFL condition for time-step (dt)...")
    # CFL condition for explicit time-integration
    dt = scaled_time_step
else:
    print("Using fixed time-step (dt)...")
    # Manual
    dt = 1e-4 / max(abs(quality),1)
print("dt = ", dt)

bspline_kernel_order = 2 # Quadratic BSpline kernel


#Added parameters for piston and particle interaction
boundary_color = 0xEBACA2 # 
board_states = ti.Vector.field(n=DIMENSIONS, dtype=float, shape=())
# board_states.from_numpy(np.array([piston_pos[0], piston_pos[1], piston_pos[2]]))
board_states[None] = [float(piston_pos[0]), float(piston_pos[1]), float(piston_pos[2])]
board_velocity = ti.Vector.field(n=DIMENSIONS, dtype=float, shape=())
board_velocity[None] = [0.0, 0.0, 0.0]
time = 0.0

#Define some parameters we would like to track
data_to_save = [] #used for saving positional data for particles 
v_data_to_save = []
bounds = [[0.0 + buffer_cells / n_grid, flume_length_3d / grid_length + buffer_cells / n_grid], [0.0 + buffer_cells / n_grid, flume_height_3d / grid_length + buffer_cells / n_grid], [0.0 + buffer_cells / n_grid, flume_width_3d / grid_length + buffer_cells / n_grid]] # For 3D

# bounds = [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]] # For 3D
vel_mean = []
vel_std = []
acc_mean = []
acc_std = []

gravity = ti.Vector.field(n=DIMENSIONS, dtype=float, shape=()) # Gravity vector, [m/s^2]

x = ti.Vector.field(DIMENSIONS, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(DIMENSIONS, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(DIMENSIONS, DIMENSIONS, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(DIMENSIONS, DIMENSIONS, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation

x.from_numpy( xyz.astype(float) ) # Load in the particle positions we made for the water and debris field
xyz = None # Clear the numpy array to save memory
material.from_numpy( material_id_numpy.astype(int) ) # Load in the material ids for the water and debris field
material_id_numpy = None # Clear the numpy array to save memory

if DIMENSIONS == 2:
    grid_tuple = (n_grid_x, n_grid_y)
elif DIMENSIONS == 3:
    grid_tuple = (n_grid_x, n_grid_y, n_grid_z)

grid_v = ti.Vector.field(DIMENSIONS, dtype=float, shape=grid_tuple)  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=grid_tuple)  # grid node interpolated mass
# grid_volume = ti.field(dtype=float, shape=grid_tuple)  # grid node interpolated volume
# grid_J = ti.field(dtype=float, shape=grid_tuple)  # grid node volume change ratio


# ti.root.place(board_states)

#Future class for different complex materials possibly??
#class material_models():
#    def __init__(self) -> None:


@ti.func
def update_material_properties(p):
    # Hardening coefficient: snow gets harder when compressed
    # Metals between h = .1-.5
    h = 1.0
    if material[p] == material_id_dict_mpm["Water"]: # 0
        h = 1.0 # No hardening for water
    elif material[p] == material_id_dict_mpm["Plastic"]: # 1 
        h = 1.0 # Do not scale elastic moduli by default
    elif material[p] == material_id_dict_mpm["Debris"]: # 2
        h = 1.0 # Do not scale elastic moduli by default
    elif material[p] == material_id_dict_mpm["Sand"] or material[p] == material_id_dict_mpm["Snow"]: # 3
        # Snow-like material, hardens when compressed, probably from Stoamkhin et al. 2013 or Klar et al.
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p])))) # Don't calc this unless used, expensive operation
    else:
        h = 1.0 # Do not scale elastic moduli by default

    mu, la = mu_0 * h, lambda_0 * h # adjust elastic moduli based on hardening coefficient

    if material[p] == material_id_dict_mpm["Water"]: # 0:  # liquid 
        mu = 0.0 # assumed no shear modulus... replace with dynamic viscosity later

    return h, mu, la


@ti.func
def neo_hookean_model(h, mu, la, F, J):
    """Due to waters near incompressibility we can treat it as a near-incompressible hyperelastic material

    Args:
        h: Hardening Coefficient
        mu: Shear modulus lame parameter
        la: First Lame parameter (bulk modulus in this context)
        F: Deformation gradient for a single particle
    Returns:

    """
    K = la 
    C = F.transpose() @ F # Left Cauchy Green Tensor
    I1 = C.trace() # Trace of Left Cauchy Green Tensor
    nhstrain = ( mu/2 ) * ( I1 - 3 ) + ( K/2 ) * ( ti.math.log(J)**2 )

    return nhstrain

@ti.func
def compute_stress_svd(p, mu, la):
    """Computing stress using svd 
    Args:
        mu: Shear modulus lame parameter
        la: First Lame parameter (bulk modulus in this context)
        p: Current particle index

    Returns:
        stress: Cauchy stress tensor
    """

    U, sig, V = ti.svd(F[p]) # Singular Value Decomposition of deformation gradient (on particle)

    #J = 1.0
    for d in ti.static(range(DIMENSIONS)):
        new_sig = sig[d, d]
        if material[p] == material_id_dict_mpm["Snow"] or material[p] == material_id_dict_mpm["Sand"]: # 2:  # Snow-like material
            # https://math.ucdavis.edu/~jteran/papers/SSCTS13.pdf
            stomakhin_critical_compression = 2.5e-2
            stomakhin_critical_stretch = 4.5e-3
            new_sig = min(max(sig[d, d], 1 - stomakhin_critical_compression), 1 + stomakhin_critical_stretch)  # Plasticity
        Jp[p] *= sig[d, d] / new_sig # stable?
        sig[d, d] = new_sig
        J *= new_sig

    if material[p] == material_id_dict_mpm["Debris"]: # 2:
        # Reconstruct elastic deformation gradient after plasticity
        F[p] = U @ sig @ V.transpose() # Singular value decomposition, good for large deformations on fixed-corotated model 
    
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, DIMENSIONS) * la * J * (
            J - 1
        )
        return stress


@ti.func
def compute_stress(mu, la, F):
    """Computes Cauchy stress tensor for a near-incompressible Neo-Hookean material 
    (can also use strain energy density function)

    Args:
        mu: Shear modulus lame parameter
        la: First Lame parameter (bulk modulus in this context)
        F: Deformation gradient for a single particle

    Returns:
        stress: Cauchy stress tensor
    """
    J = ti.math.determinant(F)
    K = la
    FinvT = F.inverse().transpose()
    
    # First Piola-Kirchhoff stress tensor (P)
    P = mu * (F - FinvT) + K * ti.math.log(J) * FinvT

    # Neo-hookean strain energy density function
    # nh_strain = neo_hookean_model( h, mu, la, F, J ) 
    
    # Cauchy stress tensor (denoted by sigma)
    stress = (1 / J) * P @ F.transpose()
    
    return stress


@ti.func
def compute_stress_jfluid(mu, la, J):
    """Computes Cauchy stress tensor for an isotropic fluid material 
    assuming Tait-Murnaghan equation of state 

    Args:
        mu: Shear modulus lame parameter
        la: First Lame parameter (bulk modulus in this context)
        J: Deformation gradient determinant for a single particle

    Returns:
        stress: Cauchy stress tensor
    """
    # J = ti.math.determinant(F[p])  #particle volume ratio = V /Vo

    # pressure = (bulk_modulus / gamma_water ) * (J - 1)
    pressure = (bulk_modulus / gamma_water ) * (ti.pow(J,-gamma_water) - 1) 
    return -pressure * ti.Matrix.identity(float, DIMENSIONS)

@ti.kernel
def substep():
    clear_grid()
    p2g()
    update_grid()
    g2p()
    # handle_piston_collisions()

@ti.func
def clear_grid():
    if ti.static(DIMENSIONS == 2):
        for i, j in grid_m:
            grid_v[i, j] = ti.Vector.zero(float, DIMENSIONS)
            grid_m[i, j] = 0
    elif ti.static(DIMENSIONS == 3):
        for i, j, k in grid_m:
            grid_v[i, j, k] = ti.Vector.zero(float, DIMENSIONS)
            grid_m[i, j, k] = 0
    else:
        raise Exception("Improper Dimensionality for Simulation Must Be 2D or 3D ")

@ti.func
def p2g():
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2] or Weights for MPM
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, DIMENSIONS) + dt * C[p]) @ F[p]

        # Hardening coefficient and Lame parameter updates
        h, mu, la = update_material_properties(p)

        # J=1 undeformed material; J<1 compressed material; J>1 expanded material
        J = ti.math.determinant(F[p])  #particle volume ratio = V /Vo


        # Reset deformation gradient to avoid numerical instability
        # if material[p] == material_id_dict_mpm["Water"]: # 0
        #     if DIMENSIONS == 2:
        #         F[p] = ti.Matrix.identity(float, DIMENSIONS) * ti.sqrt(J)
        #     elif DIMENSIONS == 3:
        #         F[p] = ti.Matrix.identity(float, DIMENSIONS) * ti.pow(J, 1/3)
                

        #Neo-hookean formulation for Cauchy stress from first Piola-Kirchoff stress
        stress = ti.Matrix.zero(float, DIMENSIONS, DIMENSIONS)
        if material[p] == material_id_dict_mpm["Water"]:
            stress = compute_stress_jfluid(mu, la, J)
        else:
            stress = compute_stress(mu, la, F[p])
                
            
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress 
        affine = stress + p_mass * C[p]
        if ti.static(DIMENSIONS == 2):
            for i, j in ti.static(ti.ndrange(3, 3)):
                # Loop over 3x3 grid node neighborhood
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
        elif ti.static(DIMENSIONS == 3):
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1] * w[k][2]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

@ti.func
def update_grid():
    if ti.static(DIMENSIONS == 2):
        for i, j in grid_m:
            if grid_m[i, j] > 0:  # No need for epsilon here
                # Momentum to velocity
                grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
                grid_v[i, j] += dt * gravity[None]   # gravity
                apply_boundary_conditions(i, j, 0)
    elif ti.static(DIMENSIONS == 3):
        for i, j, k in grid_m:
            if grid_m[i, j, k] > 0: # No need for epsilon here
                # Momentum to velocity
                grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]
                grid_v[i, j, k] += dt * gravity[None] # gravity
                apply_boundary_conditions(i, j, k)

@ti.func
def apply_boundary_conditions(i, j, k):
    
    if ti.static(DIMENSIONS == 2):
        if i < buffer_cells and grid_v[i, j][0] < 0:
            grid_v[i, j][0] = 0  # Boundary conditions
        if i > n_grid_x - buffer_cells and grid_v[i, j][0] > 0:
            grid_v[i, j][0] = 0
        if j < buffer_cells and grid_v[i, j][1] < 0:
            grid_v[i, j][1] = 0
        if j > n_grid_y - buffer_cells and grid_v[i, j][1] > 0:
            grid_v[i, j][1] = 0
        if i <= board_states[None][0] / grid_length * n_grid and grid_v[i, j][0] < board_velocity[None][0]:
            grid_v[i, j][0] = 1.0 * board_velocity[None][0]
    elif ti.static(DIMENSIONS == 3):
        if i < buffer_cells and grid_v[i, j, k][0] < 0:
            grid_v[i, j, k][0] = 0
        if i > n_grid_x - buffer_cells and grid_v[i, j, k][0] > 0:
            grid_v[i, j, k][0] = 0
        if j < buffer_cells and grid_v[i, j, k][1] < 0:
            grid_v[i, j, k][1] = 0
        if j > n_grid_y - buffer_cells and grid_v[i, j, k][1] > 0:
            grid_v[i, j, k][1] = 0
        if k < buffer_cells and grid_v[i, j, k][2] < 0:
            grid_v[i, j, k][2] = 0
        if k > n_grid_z - buffer_cells and grid_v[i, j, k][2] > 0:
            grid_v[i, j, k][2] = 0
        if i > flume_length_3d / grid_length * n_grid - buffer_cells and grid_v[i, j, k][0] > 0:
            grid_v[i, j, k][0] = 0
        if j > flume_height_3d / grid_length * n_grid - buffer_cells and grid_v[i, j, k][1] > 0:
            grid_v[i, j, k][1] = 0
        if k > flume_width_3d / grid_length * n_grid - buffer_cells and grid_v[i, j, k][2] > 0:
            grid_v[i, j, k][2] = 0
        if i <= board_states[None][0] / grid_length * n_grid and grid_v[i, j, k][0] < board_velocity[None][0]:
            grid_v[i, j, k][0] = 1.0 * board_velocity[None][0]
    # piston_pos_current = board_states[None][0]

    # piston_time_stdev = (piston_scale_factor) * 0.707106781187 # (SF / 100) / sqrt(2)
    # piston_time_variance = piston_time_stdev*piston_time_stdev
    # t = time - piston_wait_time
    # piston_pos_current = ti.max(4*dx, ti.min(0.5 * piston_amplitude * ((erf_approx(((t - piston_time_mean)) ) + 1.0) ), 4*dx + piston_amplitude))
    # piston_vel_current = (piston_amplitude / piston_time_stdev) * (ti.exp(-((time - piston_wait_time - piston_time_mean)/(piston_time_stdev*1.41421356237))*((time-piston_wait_time-piston_time_mean)/(piston_time_stdev * 1.41421356237))/2)) * 0.398942280401
    # if (verbose == True):
        # print("Piston Velocity: ", piston_vel_current)
    # print("Time: ", time)
    # print("Piston Position: ", piston_pos_current)
    # print("Piston init Position: ", piston_start_x)
    # print("Piston Velocity: ", piston_vel_current)


@ti.func
def g2p():
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, DIMENSIONS)
        new_C = ti.Matrix.zero(float, DIMENSIONS, DIMENSIONS)
        if ti.static(DIMENSIONS == 2):
            for i, j in ti.static(ti.ndrange(3, 3)):
                # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = grid_v[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        elif ti.static(DIMENSIONS == 3):
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                # loop over 3x3x3 grid node neighborhood
                dpos = ti.Vector([i, j, k]).cast(float) - fx
                g_v = grid_v[base + ti.Vector([i, j, k])]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection


# @ti.func
# def handle_piston_collisions():
#     for p in x:
#         # Apply piston force based on Hooke's law
#         # piston_pos_current = board_states[None][0]
#         piston_pos_current = ti.max(4*dx, ti.min(0.5 * piston_amplitude * ((erf_approx(((time - piston_time_mean)) ) + 1.0) ), 4*dx + piston_amplitude))
#         if x[p][0] < piston_pos_current:
#             # Using separable contact, i.e. water doesn't stick if not being pushed
#             displacement_into_piston = ti.max(piston_pos_current - x[p][0], 0.0)
#             piston_spring_constant = p_mass / dt  # Assume a 1.0 kg mass 
#             force = ti.max(piston_spring_constant * displacement_into_piston, 0.0)  # Hooke's law: F = k * x
#             piston_escape_velocity = 1 * force / p_mass * dt  # v = F / m * dt
#             piston_escape_velocity = ti.min(piston_escape_velocity, max_vel)  # Cap the velocity to prevent instability
#             v[p][0] = ti.max(v[p][0], piston_escape_velocity)  # Stop the particle from moving into the piston

# @ti.func
def erf_approx(erf_x):
    """Needed an approximation to the gauss error function (math lib doesnt work with taichi)
    
    From: https://en.wikipedia.org/wiki/Error_function
    """
    # Approximation constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    erf_p  =  0.3275911

    # Save the sign of x
    sign = 1
    if erf_x < 0:
        sign = -1
    erf_x = abs(erf_x)

    # Abramowitz and Stegun formula 
    erf_t = 1.0 / (1.0 + erf_p * erf_x)
    erf_y = 1.0 - (((((a5 * erf_t + a4) * erf_t) + a3) * erf_t + a2) * erf_t + a1) * erf_t * math.exp(-erf_x * erf_x)

    return sign * erf_y


def move_board_solitary():
    t = time - piston_wait_time 
    b = board_states[None]
    bv = board_velocity[None]
    # b[1] += dt  # Adjusting for the coordinate frame

    # if b[1] >= 2 * piston_period:
    #     b[1] = 0


    piston_time_variance = piston_time_stdev * piston_time_stdev
    b[0] = 0.5 * piston_amplitude * ((math.erf((t - piston_time_mean) / (piston_time_stdev) * 0.707106781187 ) + 1.0) ) + piston_pos[0]  #Soliton wave
    # b[0] = 0.5 * piston_amplitude * ((erf_approx((t - piston_time_mean - 1e-2)) + 1.0) ) + 4 * dx  #Soliton wave
    bv[0] = (piston_amplitude / piston_time_stdev) * (math.exp(-((t - piston_time_mean)/(piston_time_stdev))*((t - piston_time_mean)/(piston_time_stdev)) * 0.5)) * 0.398942280401
    
    # b[0] += 0.
    # bv[0] += 0.
    # print("Piston Velocity: ", piston_vel_current)
    # Ensure the piston stays within the boundaries
    # b[0] = ti.max(0.0, ti.min(b[0], piston_pos[0] + piston_amplitude))
    
    # Store the updated state back to the field
    board_states[None] = b
    board_velocity[None] = bv


@ti.kernel
def reset():
    for i in range(n_particles):
        if ti.static(DIMENSIONS == 2):
            v[i] = [0.0, 0.0]
            F[i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            Jp[i] = 1.0
            C[i] = ti.Matrix.zero(float, DIMENSIONS, DIMENSIONS)
        elif ti.static(DIMENSIONS == 3):
            v[i] = [0.0, 0.0, 0.0]
            F[i] = ti.Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            Jp[i] = 1.0
            C[i] = ti.Matrix.zero(float, DIMENSIONS, DIMENSIONS)
        
    if ti.static(DIMENSIONS == 2):
        board_states[None] = [float(piston_pos[0]), 0.0]  # Initial piston position
    elif ti.static(DIMENSIONS == 3):
        board_states[None] = [float(piston_pos[0]), 0.0, 0.0]  # Initial piston position
    
    # water_ratio_denominator = 64
    # group_size = n_particles // water_ratio_denominator
    # basin_row_size = int(ti.floor((flume_length_3d) / grid_length * n_grid * particles_per_dx))
    # debris_row_size = int(ti.floor(4 * particles_per_dx))
   
    # for i in range(n_particles):
        
    #     row_size = basin_row_size 
    #     # j = i // row_size
    #     water_ratio_numerator = water_ratio_denominator - 1
    #     n_water_particles = water_ratio_numerator * group_size
    #     if i < n_water_particles:
    #         if ti.static(DIMENSIONS == 2):
    #         # ppc = 4
    #         # x is: scaled starting pos * position spacing * even distribution along x
    #         # y is: Base offset * position spacing * how many rows placed for spacing
    #             x[i] = [
    #                 # ti.random() * 0.8 + 0.01 * (i // group_size),  # Fluid particles are spread over a wider x-range
    #                 # ti.random() * 0.1 + 0.01 * (i // group_size)  # Fluid particles are spread over a wider y-range
    #                 (piston_start_x * grid_length) + (dx * particle_spacing_ratio) * (i % row_size),  # Fluid particles are spread over a wider x-range
    #                 (2.5 * dx) + (dx * particle_spacing_ratio) * (i // row_size)  # Fluid particles are spread over a wider y-range
    #             ]
    #         if ti.static(DIMENSIONS == 3):
    #             row_size_x = i % row_size
    #             row_size_y = ((i // row_size) % int(ti.floor(((max_water_depth_tsunami / (dx) - 3)* particles_per_dx)) ))
    #             row_size_z = ((i // row_size) // int(ti.floor(((max_water_depth_tsunami / (dx) - 3)* particles_per_dx)) )) # will later add checks for init within the flume (i.e. dont init outside of it )
    #             row_size_z = (i % row_size_y) // int(ti.floor((flume_width_3d / (dx * particle_spacing_ratio)) - 3))
    #             row_size_z = int(ti.floor((flume_width_3d / (dx * particle_spacing_ratio)) - 3))
    #             # xyz numpy to taichi
    #             # x[i] = [xyz[None,0], xyz[None,1], xyz[None,2]]
                
    #             x[i] = [
    #                 (piston_start_x * grid_length) + (dx * particle_spacing_ratio) * row_size_x,  # x-position
    #                 (2.5 * dx) + (dx * particle_spacing_ratio) * row_size_y, # y-position
    #                 (2.5 * dx) + (dx * particle_spacing_ratio) * row_size_z,  # z-position
    #             ]
    #         material[i] = 0  # fluid

    #     else:

    #         # Choose shape
    #         shape = 0
    #         id = i % (n_water_particles)
    #         row_size = debris_row_size
    #         block_size = row_size**2
    #         debris_particle_x = ti.min(grid_length_x, (4*dx ) + (grid_length * (piston_start_x + piston_travel_x)) + (dx * particle_spacing_ratio) * ((id % row_size**2) % row_size) + grid_length * (16 * dx / grid_length) * (id // (row_size**2)))
    #         if shape == 0:
    #             if ti.static(DIMENSIONS == 2):
    #                     debris_particle_y = ti.min(grid_length_y, (4*dx) + (dx * (1 + particle_spacing_ratio * n_water_particles // basin_row_size)) + (dx * particle_spacing_ratio * ((id % row_size**2) // row_size)))
    #                     x[i] = [
    #                         debris_particle_x,  # Block particles are confined to a smaller x-range
    #                         debris_particle_y   # Block particles are confined to a smaller y-range
    #                     ]
    #             elif ti.static(DIMENSIONS == 3):
    #                 debris_particle_y = ti.min(max_water_depth_tsunami + ((2.5+1)*dx), ((2.5+1)*dx) + (max_water_depth_tsunami) + (dx * particle_spacing_ratio * ((id % row_size**2) // row_size)))
    #                 debris_particle_z = ti.min(flume_width_3d + (2.5*dx), (flume_width_3d / 2.0) + (2.5*dx) + (dx * (1 + particle_spacing_ratio * n_water_particles // basin_row_size)) + (dx * particle_spacing_ratio * ((id % row_size**2) // row_size)))
    #                 x[i] = [
    #                     debris_particle_x,  # Block particles are confined to a smaller x-range
    #                     debris_particle_y,   # Block particles are confined to a smaller y-range
    #                     debris_particle_z
    #                 ]
    #         material[i] = 1  # Fixed-Corotated Hyper-elastic debris (e.g. for simple plastic, metal, rubber)
     

def save_metadata(file_path):
    """Save metadata.json to file
    Args:
        file_path: the path to save the metadata (**Automatically retrieved by system**)
        bounds: The boundaries for plotting the animated simulation
        sequence_length: The number of time steps taken
        DIMENSIONS: The dimensionality of the simulation (i.e. 2d or 3d)
        time_delta: Defined as 1 / frames per second of the simulation (typically 20-30 fps)
        dx: Change in grid sizing defined as the grid_length / # of Grids
        dt: Time rate of change using CFL for simulation stability (dt = CFL * dx / max_vel)

    ** NOTE: GNS currently does not have use for variables "critical_time_step": dt or "dx": dx. **
       
       Returns:
        None
    """
    #Using a list for each time step for formatting
    global v_data_to_save
    vel = np.stack(v_data_to_save,axis=0) / grid_length # Scale velocity data to 1x1x1 domain for GNS
    vel_diff = np.diff(vel, axis=0) #computing acceleration along the time dependant axis
    
    #Define meta data dictionary from trajectories and timesteps
    vel_mean = np.nanmean(vel, axis=(0, 1))
    vel_std = np.nanstd(vel, axis=(0, 1)) #standard deviation of velocity
    acc_mean = np.nanmean(vel_diff, axis=(0, 1)) #mean acceleration from velocity
    acc_std = np.nanstd(vel_diff, axis=(0, 1))  #standard deviation of acceleration from velocity 
   
    # Convert numpy types to native Python types
    vel_mean = [float(x) for x in vel_mean]
    vel_std = [float(x) for x in vel_std]
    acc_mean = [float(x) for x in acc_mean]
    acc_std = [float(x) for x in acc_std]
    
    # Rough estimate of the bspline interaction range between initial particles
    connectivity_amplifier = 1.5
    bspline_max_reach = 1.5 # [Cell Units]
    downsampled_connectivity_radius = (2 * dx * connectivity_amplifier * bspline_max_reach) / (downsampling_ratio**(1/DIMENSIONS))
    downsampled_connectivity_radius = float(downsampled_connectivity_radius / grid_length) # Scale to 1x1x1 domain for GNS
    
    # Assume scaled already
    bounds_metadata = []
    for i in range(DIMENSIONS):
        bounds_i = [float(bound_ii) for bound_ii in bounds[i]]
        bounds_metadata.append(bounds_i)
    
    for i in range(DIMENSIONS):
        while (bounds_metadata[i][0] < 0.0 or bounds_metadata[i][0] > 1.0):
            print("Scale bounds to 0 to 1 range")
            bounds_metadata[i][0] = float(abs(bounds_metadata[i][0]) / grid_length)
            
        while (bounds_metadata[i][1] > 1.0 or bounds_metadata[i][1] < 0.0):    
            print("Scale bounds to 0 to 1 range")
            bounds_metadata[i][1] = float(abs(bounds_metadata[i][1]) / grid_length)
        
        if (bounds_metadata[i][0] == bounds_metadata[i][1]):
            print("Assume that the bounds were centered around the origin, so we shift the to the right into 0 to 1 range")
            bounds_metadata[i][0] = 0.5 - min(0.5, bounds_metadata[i][0])
            bounds_metadata[i][1] = 0.5 + min(0.5, bounds_metadata[i][1])
            
        if (bounds_metadata[i][0] > bounds_metadata[i][1]):
            print("Swap the bounds if they are in the wrong order, i.e. [1, 0] -> [0, 1]")
            bounds_temp = float(bounds_metadata[i][0])
            bounds_metadata[i][0] = float(bounds_metadata[i][1])
            bounds = bounds_temp
            
        print("bounds for metadata to GNS: ", bounds_metadata)
    
    # Formatting enforced
    metadata = {
        "bounds": bounds_metadata,
        "sequence_length": sequence_length, 
        "default_connectivity_radius": downsampled_connectivity_radius, 
        "dim": DIMENSIONS, 
        "dt": 1.0 / fps, 
        "dx": dx,
        "critical_time_step": dt,
        "vel_mean": vel_mean, #[5.123277536458455e-06, -0.0009965205918140803], 
        "vel_std": vel_std, #[0.0021978993231675805, 0.0026653552458701774], 
        "acc_mean": acc_mean, #[5.237611158734309e-07, 2.3633027988858656e-07], 
        "acc_std": acc_std, #[0.0002582944917306106, 0.00029554531667679154]
    }
    
    print("Metadata: ", metadata)
    
    # Write metadata to a JSON file
    with open(os.path.join(file_path, 'metadata.json'), 'w') as file:
        json.dump(metadata, file)
        print("Metadata Saved!\n")

    
def save_simulation():
    """Save train.npz, test.npz,or valid.npz to file
    Args:
        None
    Returns:
        None
    """
    global data_designation
    global data

    # Define file_path to save to data, models, rollout folder. Located in directory of this file script

    system = platform.system().lower()

    if system == 'linux':
        file_path = "./Flume/dataset"
    elif system == 'darwin':  # 'Darwin' is the system name for macOS
        file_path = "Flume/dataset"
        #file_path = "/Users/" + user_name + "/code-REU/Physics-Informed-ML/dataset/"
    elif system == 'windows':
        file_path = "./Flume/dataset"
    else:
        file_path = "./Flume/dataset"

    save_relative_to_cwd = True
    ROLLOUT_PATH="./Flume/rollout"
    MODEL_PATH="./Flume/models"
    if save_relative_to_cwd:
         cwd_path = os.getcwd()
         file_path = os.path.join(cwd_path, file_path)
         ROLLOUT_PATH = os.path.join(cwd_path, ROLLOUT_PATH)
         MODEL_PATH = os.path.join(cwd_path, MODEL_PATH)

    # Ensuring the directories exist within the cwd
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(ROLLOUT_PATH):
        os.makedirs(ROLLOUT_PATH)
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)


    material_numpy = material.to_numpy()
    mat_data_tmp = np.where(material_numpy == material_id_dict_mpm["Water"], material_id_dict_gns["Water"] + (0 * material_numpy), material_numpy)

    mat_data = np.asarray(mat_data_tmp, dtype=object)
    pos_data = np.stack(data_to_save, axis=0)

    # Perform downsampling for GNS
    if downsampling:
        downsampled_mat_data = mat_data[::downsampling_ratio]
        downsampled_data = pos_data[:,::downsampling_ratio,:]


    #check version of numpy >= 1.22.0
    # Newer versions of numpy require the dtype to be explicitly set to object, I think, for some python versions
    # Should add a check for the python version as well
    
    if (np.version.version >= '1.23.5'):
        print("Using numpy version (>= 1.23.5), may require alternative approach to save npz files (e.g. dtype=object): ", np.version.version)
        pos_data = np.array(np.stack(np.asarray(downsampled_data, dtype=object), axis=0), dtype=object)
        mat_data = np.asarray(downsampled_mat_data, dtype=object)
    else:
        print("Warning: Using numpy version: ", np.version.version)
        pos_data = np.array(np.stack(downsampled_data, axis=0), dtype=np.float32)
        mat_data = np.asarray(downsampled_mat_data, dtype=object)
        # np.array(material_data.tolist()) 
    
    print("pos_data: ", pos_data.shape)
    print("mat_data: ", mat_data.shape)
    simulation_data = {
        'simulation_0': (
            pos_data,
            mat_data
        )
    }
        
    if data_designation.lower() in ("r", "rollout", "test"):
        # Should clarify the difference in naming between test and rollout
        output_file_path = os.path.join(file_path, "test.npz")
        np.savez_compressed(f'{file_path}/test.npz', **simulation_data)

    elif data_designation.lower() in ("t", "train"):
        output_file_path = os.path.join(file_path, "train.npz")
        np.savez_compressed(f'{file_path}/train.npz', **simulation_data)
        save_metadata(file_path)

    elif data_designation.lower() in ("v", "valid"):
        output_file_path = os.path.join(file_path, "valid.npz")
        np.savez_compressed(f'{file_path}/valid.npz', **simulation_data) # Proper 
        
    else:
        output_file_path = os.path.join(cwd_path, "unspecified_sim_data.npz")
        np.savez_compressed("unspecified_sim_data2.npz", **simulation_data)
        #np.savez_compressed('simulation_data_kwargs.npz', pos_data=downsampled_data, material_ids=downsampled_mat_data)
        # Save to HDF5
        #with h5py.File(f'{cwd_path}/unspecified_sim_data.h5', 'w') as f:
        #    f.create_dataset('pos_data', data=downsampled_data)
        #    f.create_dataset('material_ids', data=downsampled_mat_data)
        
    print("Simulation Data Saved to: ", file_path)

# Define a Taichi field to store the result
    
@ti.kernel
def create_flume_vertices():
    flume_vertices[0] = ti.Vector([0, 0, 0])
    flume_vertices[1] = ti.Vector([grid_length, 0, 0])
    flume_vertices[2] = ti.Vector([grid_length, 0, flume_width_3d])
    flume_vertices[3] = ti.Vector([0, 0, flume_width_3d])
    flume_vertices[4] = ti.Vector([0, flume_height_3d, 0])
    flume_vertices[5] = ti.Vector([grid_length, flume_height_3d, 0])
    flume_vertices[6] = ti.Vector([grid_length, flume_height_3d, flume_width_3d])
    flume_vertices[7] = ti.Vector([0, flume_height_3d, flume_width_3d])

@ti.kernel
def create_flume_indices():
    # Bottom face
    bottom[0], bottom[1], bottom[2] = 0, 1, 2
    bottom[3], bottom[4], bottom[5] = 0, 2, 3

    # Side faces
    sidewalls[0], sidewalls[1], sidewalls[2] = 0, 4, 5
    sidewalls[3], sidewalls[4], sidewalls[5] = 0, 5, 1

    # We want this face transparent to see inside the Flume
    #sidewalls[6], sidewalls[7], sidewalls[8] = 0, 1, 2
    #sidewalls[9], sidewalls[10], sidewalls[11] = 0, 2, 3
    
    # Front and Back faces
    backwall[0], backwall[1], backwall[2] = 1, 5, 6
    backwall[3], backwall[4], backwall[5] = 1, 6, 2
    frontwall[0], frontwall[1], frontwall[2] = 3, 7, 4
    frontwall[3], frontwall[4], frontwall[5] = 3, 4, 0

# def copy_to_field(source: ti.types.ndarray, target: ti.template):
@ti.kernel
def copy_to_field(source: ti.types.ndarray(), target: ti.template()):
    for i in range(source.shape[0]):
        for j in ti.static(range(DIMENSIONS)):
            target[i][j] = source[i, j]

def render_3D():
    #camera.position(grid_length*1.2, flume_height_3d*10, flume_width_3d*8) #Actual Camera to use
    #camera.position(grid_length*1.5, flume_height_3d*4, flume_width_3d*6) # 50m flume camera
    camera.position(grid_length*1.5, flume_height_3d*4, flume_width_3d*.5) # Front View

    camera.lookat(grid_length/2, flume_height_3d/2, flume_width_3d/2)
    camera.up(0, 1, 0)
    camera.fov(60)
    scene.set_camera(camera)

    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(grid_length/2, flume_height_3d*2, flume_width_3d*2), color=(1, 1, 1))

    # Render the flume
    # Render the bottom face
    scene.mesh(flume_vertices, bottom, color=bottom_color)

    # Render each face separately (if only taichi supported slicing)
    scene.mesh(flume_vertices, backwall, color=front_back_color)
    scene.mesh(flume_vertices, sidewalls, color=side_color)
    scene.mesh(flume_vertices, frontwall, color=front_back_color)
    #scene.particles(x, radius=0.002*grid_length, color=(50/255, 92/255, 168/255))
    # Scale the color palette for 3d by how many materials we want
    colors_np = np.array([palette[material[i]] for i in range(n_particles)], dtype=np.float32)
    colors_field = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    # Copy data to Taichi fields
    copy_to_field(colors_np, colors_field)

    scene.particles(x, per_vertex_color=colors_field, radius=0.002*grid_length)
    canvas.scene(scene)

#Simulation Prerequisites 
data_designation = str(input('What is the output particle data for? Select: Rollout(R), Training(T), Valid(V) [Waiting for user input...] --> '))
# sequence_length = int(input('How many time steps to simulate? --> ')) 
fps = int(input('How many frames-per-second (FPS) to output? [Waiting for user input...] -->'))
sequence_length = int(input('How many seconds to run this simulations? [Waiting for user input...] --> ')) * fps # May want to provide an FPS input 





gui_res = min(1024, n_grid) # Set the resolution of the GUI
gui_res_base = 1024
gui_res_ratio_x = int(1 / grid_ratio_x)
gui_res_ratio_y = int(1 / grid_ratio_y)
gui_res_ratio_z = int(1 / grid_ratio_z)



if DIMENSIONS == 2:
    palette = [0x2389da, 0xED553B, 0x068587, 0x6D214F]
    gravity[None] = [0.0, -9.80665] # Gravity in m/s^2, this implies use of metric units
    gui_background_color_white = 0xFFFFFF # White or black generally preferred for papers / slideshows, but its up to you
    gui_background_color_taichi= 0x112F41 # Taichi default background color, may be easier on the eyes  
    gui_res = (2048, int(2048 * grid_ratio_y))
    gui = ti.GUI("Digital Twin of the NSF OSU LWF Facility - Tsunami Debris Simulation in MPM - 2D", 
                res=gui_res, background_color=gui_background_color_white)

elif DIMENSIONS == 3 and use_vulkan_gui:
    palette = [(35/255, 137/255, 218/255), 
           (237/255, 85/255, 59/255), 
           (6/255, 133/255, 135/255), 
           (109/255, 33/255, 79/255)]
    gravity[None] = [0.0, -9.80665, 0.0] # Gravity in m/s^2, this implies use of metric units
    # Initialize flume geometry
    create_flume_vertices()
    create_flume_indices()

    # Initialize the GUI
    gui_res = (1024, 1024)
    gui = ti.ui.Window("Digital Twin of the NSF OSU LWF Facility - Tsunami Debris Simulation in MPM - 3D", res = (gui_res, gui_res))
    canvas = gui.get_canvas()
    scene = gui.get_scene()
    camera = ti.ui.Camera()

elif DIMENSIONS == 3 and not use_vulkan_gui:
    gui_res_for_multi_viewport = (int(1.0*(flume_length_3d + flume_length_3d) / grid_length * gui_res_base), int(1.75*(grid_length_y + grid_length_z) / grid_length * gui_res_base)) 
    palette = [0x2389da, 0xED553B, 0x068587, 0x6D214F]
    gravity[None] = [0.0, -9.80665, 0.0] # Gravity in m/s^2, this implies use of metric units
    gui_background_color_white = 0xFFFFFF # White or black generally preferred for papers / slideshows, but its up to you
    gui_background_color_taichi= 0x112F41 # Taichi default background color, may be easier on the eyes  
    gui_res = (2048, int(2048 * grid_ratio_z))
    gui = ti.GUI("Digital Twin of the NSF OSU LWF Facility - Tsunami Debris Simulation in MPM - 3D - Side-View", 
                res=gui_res_for_multi_viewport, background_color=gui_background_color_white)
    
# Saving Figures of the simulation (2D only so far)
base_frame_dir = './Flume/figures/'
os.makedirs(base_frame_dir, exist_ok=True) # Ensure the directory exists
frame_paths = []

reset() # Reset sim and initialize particles

for frame in range(sequence_length):
    # for s in range(int(2e-3 // dt)): # Will need to double-check the use of 2e-3, dt, etc.
    for s in range(int((1.0/fps) // dt)): 
        move_board_solitary()
        substep()
        time += dt # Update time by dt so that the time used in move_board_solitary() is accurate, otherwise the piston moves only once every frame position-wise which causes instabilities


    print(f't = {round(time,3)}')


    print(f'Piston Position x = {round(board_states[None][0],5)}')
    print(f'Piston Velocity V_x = {round(board_velocity[None][0],5)}')

    
    #Change to tiachi fields probably
    data_to_save.append(x.to_numpy())
    v_data_to_save.append(v.to_numpy())
    
    clipped_material = np.clip(material.to_numpy(), 0, len(palette) - 1) #handles error where the number of materials is greater len(palette)
    # print("TestHex: ", int('0x000000',0))
    # print(cm.plasma(x.to_numpy()[:, 1] / flume_height_3d)[:,:3].max())
    # Elevation_palette = []
    # for rgba_tuple in Elevation_img:
    
    #     Elevation_palette.append(int('0x' + ''.join(f'{rgb_component:02X}' for rgb_component in rgba_tuple[:3])))


    # visualize_variable = "Velocity"
    # if visualize_variable == "Elevation":
    #     Elevation_img = (cm.plasma(x.to_numpy()[:, 1] / Elevation_vmax)[:,:3] * 255)
    #     chosen_palette = [ (int('0x' + ''.join(f'{rgb_component:02X}' for rgb_component in rgba_tuple[:3]), 0)) for rgba_tuple in Elevation_img] 
    # elif visualize_variable == "Velocity":
    #     Velocity_vmax = 0.1
    #     Velocity_img = (cm.plasma(np.sqrt(v.to_numpy()[:, 0]**2 + v.to_numpy()[:, 1]**2 + v.to_numpy()[:,2]**2) / Velocity_vmax)[:,:3] * 255).astype(np.uint8)
    #     chosen_palette = Velocity_img

    Elevation_vmax = 4.0
    # Elevation_img = (cm.plasma(x.to_numpy()[:, 1] / Elevation_vmax)[:,:3] * 255).astype(np.uint8)
    # Elevation_palette = [ (int('0x' + ''.join(f'{rgb_component:02X}' for rgb_component in rgba_tuple[:3]), 0)) for rgba_tuple in Elevation_img] 

    Velocity_vmax = 2.0
    Velocity_img = ((cm.viridis(np.sqrt(v.to_numpy()[:n_particles_water, 0]**2 + v.to_numpy()[:n_particles_water, 1]**2 + v.to_numpy()[:n_particles_water,2]**2) / Velocity_vmax)[:,:3]) * 255).astype(np.uint8)
    Velocity_img = np.concatenate((Velocity_img, ((cm.magma(np.sqrt(v.to_numpy()[n_particles_water:(n_particles_water+n_particles_debris_group), 0]**2 + v.to_numpy()[n_particles_water:(n_particles_water+n_particles_debris_group), 1]**2 + v.to_numpy()[n_particles_water:(n_particles_water+n_particles_debris_group),2]**2) / Velocity_vmax)[:,:3]) * 255).astype(np.uint8)), axis=0)
    
    Velocity_palette = [ (int('0x' + ''.join(f'{rgb_component:02X}' for rgb_component in rgba_tuple[:3]), 0)) for rgba_tuple in Velocity_img] 

    Pressure_vmax = max_water_depth_tsunami * 36000.0
    Pressure_img = (cm.turbo(np.maximum(0.0,(bulk_modulus / gamma_water) * (np.linalg.det(F.to_numpy()[:])**(-gamma_water) - 1) / (2 * Pressure_vmax) + 0.5))[:,:3] * 255).astype(np.uint8)
    # Reverse the color map for pressure
    
    Pressure_palette = [ (int('0x' + ''.join(f'{rgb_component:02X}' for rgb_component in rgba_tuple[:3]), 0)) for rgba_tuple in Pressure_img]
    chosen_palette = Velocity_palette
    
    if DIMENSIONS == 2:
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
 
        # vel_mag = ([:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
        Jp_img = cm.plasma(Jp / 0.15)
        gui.circles(
            x.to_numpy() / grid_length,
            radius=1.0,
            palette=palette,
            palette_indices= clipped_material,
        )

        # gui.set_image(Elevation_img)

        # Render the moving piston
        piston_pos_current = board_states[None][0]
        piston_draw = np.array([board_states[None][0] / grid_length * gui_res_ratio[0], board_states[None][1] / grid_length * gui_res_ratio[1]])
        
        #print(piston_pos)
        gui.line(
            [piston_draw[0], 0.0], [piston_draw[0], 1.0],
            color=boundary_color,
            radius=2
        )
        gui.line(
            [0.0, grid_ratio_y], [grid_ratio_x, grid_ratio_y],
            color=boundary_color,
            radius=2
        )
        
        p1 = [3/n_grid, 3/n_grid]
        p2 = [(n_grid-6)/n_grid, (n_grid-6)/n_grid]
        while gui.running:
            gui.rect(topleft=p1, bottomright=p2, color=0x000000)
    elif DIMENSIONS == 3:
        # DO NOT USE MATPLOTLIB FOR 3D RENDERING
        # Update the scene with particle positions

        if (use_vulkan_gui):

            render_3D() # Show window is handled below 
            # gui.update()
            for event in gui.get_events(ti.ui.PRESS):
                if event.key == ti.ui.ESCAPE:
                    break

        else:
            if gui.get_event(ti.GUI.PRESS):
                if gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                    break

            gui_res_ratio = np.array([gui_res_ratio_x, gui_res_ratio_y, gui_res_ratio_z])
            viewport_buffer = 1.025 # Buffer ratio to ensure each flume viewport is visible and separated
            view_style = "both"
            if view_style != "both":
                if view_style == "top":
                    view_slices = [0,2]
                elif view_style == "side":
                    view_slices = [0,1]
                else:
                    view_slices = [0,1]
                gui.circles(
                    x.to_numpy()[:,view_slices] / grid_length * gui_res_ratio[view_slices],
                    radius=1.0,
                    palette = chosen_palette,
                    palette_indices = [palette_idx for palette_idx in range(len(chosen_palette))]
                )
            else:
                gui.circles(
                    x.to_numpy()[:,[0,1]] / grid_length * gui_res_ratio[[0,1]],
                    radius=1.5,
                    palette = chosen_palette,
                    palette_indices = [palette_idx for palette_idx in range(len(chosen_palette))]
                )
                gui.circles(
                    x.to_numpy()[:,[2,1]] / grid_length * gui_res_ratio[[0,1]] + np.array([(flume_length_3d * viewport_buffer) / grid_length * gui_res_ratio[0], 0.0]),
                    radius=1.5,
                    palette = chosen_palette,
                    palette_indices = [palette_idx for palette_idx in range(len(chosen_palette))]
                )
                gui.circles(
                    x.to_numpy()[:,[0,2]] / grid_length * gui_res_ratio[[0,1]] + np.array([0.0, (flume_height_3d * viewport_buffer) / grid_length * gui_res_ratio[1]]),
                    radius=1.5,
                    palette = chosen_palette,
                    palette_indices = [palette_idx for palette_idx in range(len(chosen_palette))]
                )
                
                chosen_palette = Pressure_palette
                
                gui.circles(
                    x.to_numpy()[:,[0,1]] / grid_length * gui_res_ratio[[0,1]] + np.array([0.0, (flume_height_3d * (viewport_buffer + viewport_buffer)) / grid_length * gui_res_ratio[1]]),
                    radius=1.5,
                    palette = chosen_palette,
                    palette_indices = [palette_idx for palette_idx in range(len(chosen_palette))],
                )
                gui.circles(
                    x.to_numpy()[:,[2,1]] / grid_length * gui_res_ratio[[0,1]] + np.array([(flume_length_3d * (viewport_buffer)) / grid_length * gui_res_ratio[0], (flume_height_3d * (viewport_buffer + viewport_buffer)) / grid_length * gui_res_ratio[1]]),
                    radius=1.5,
                    palette = chosen_palette,
                    palette_indices = [palette_idx for palette_idx in range(len(chosen_palette))]
                )
                gui.circles(
                    x.to_numpy()[:,[0,2]] / grid_length * gui_res_ratio[[0,1]] + np.array([0.0, (flume_height_3d * (viewport_buffer + viewport_buffer + viewport_buffer)) / grid_length * gui_res_ratio[1]]),
                    radius=1.5,
                    palette = chosen_palette,
                    palette_indices = [palette_idx for palette_idx in range(len(chosen_palette))]
                )


            # Render the moving piston
            # piston_pos_current = board_states[None][0]
            
            #print(piston_pos)
            if view_style == "top" or view_style == "both":
                piston_draw = np.array([board_states[None][0] / grid_length , flume_width_3d/ grid_length ])
                gui.line(
                    [piston_draw[0], 0.0], [piston_draw[0], flume_width_3d / grid_length * gui_res_ratio_z],
                    color=boundary_color,
                    radius=2
                )
                gui.line(
                    [0.0, grid_ratio_z*gui_res_ratio_z], [grid_ratio_x, grid_ratio_z*gui_res_ratio_z],
                    color=boundary_color,
                    radius=2
                )

            if view_style == "side" or view_style == "both":
                piston_draw = np.array([board_states[None][0] / grid_length , flume_height_3d/ grid_length ])

                gui.line(
                    [piston_draw[0], 0.0], [piston_draw[0], flume_height_3d / grid_length * gui_res_ratio_y],
                    color=boundary_color,
                    radius=2
                )
                gui.line(
                    [0.0, grid_ratio_y*gui_res_ratio_y], [grid_ratio_x, grid_ratio_y*gui_res_ratio_y],
                    color=boundary_color,
                    radius=2
                )

            p1 = [3/n_grid, 3/n_grid]
            p2 = [(n_grid-6)/n_grid, (n_grid-6)/n_grid]
            gui.rect(topleft=p1, bottomright=p2, color=0x000000)

    frame_filename = f'frame_{frame:05d}.png'
    frame_path = os.path.join(base_frame_dir, frame_filename)

    if output_png and output_gui:
        try:
            gui.show(frame_path)
        except Exception as e:
            print(f"Error showing frame: {e}")
            # Fallback to imwrite
            try:
                tools.imwrite(x.to_numpy(), frame_path)
                frame_paths.append(frame_path)
            except Exception as e:
                print(f"Error writing frame: {e}")
        else:
            frame_paths.append(frame_path)
    elif output_png and not output_gui:
        try:
            tools.imwrite(x.to_numpy(), frame_path)
            frame_paths.append(frame_path)
        except Exception as e:
            print(f"Error writing frame: {e}")
    elif output_gui and not output_png:
        gui.show()
    else:
        print("WARNING - No output method selected, frame not saved or displayed...")
    if not output_gui:
        continue

# Check if there are frames to create a GIF
if frame_paths:
    gif_path = f"./Flume/simulation_{n_particles}.gif"
    try:
        with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)
        print(f"GIF created at {gif_path}")
    except Exception as e:
        print(f"Error creating GIF: {e}")
    
#Prep for GNS input
save_simulation()

#if using save_sim.py script
#ss.save_sim_data(data_designation, data_to_save, v_data_to_save, material, 
 #                bounds, sequence_length, DIMENSIONS, time_delta, dx, dt)