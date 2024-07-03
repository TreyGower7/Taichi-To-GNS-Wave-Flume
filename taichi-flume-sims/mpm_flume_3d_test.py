#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Added above lines for external execution of the script
import taichi as ti
import numpy as np
import platform # For getting the operating system name, taichi may already have something for this
import os
import json
import math
from taichi import tools
import imageio
import save_sim as ss
import time as T


ti.init(arch=ti.gpu)  # Try to run on GPU

DIMENSIONS = 2 # DIMENSIONS, 2D or 3D
output_gui = True # Output to GUI window (original, not GGUI which requires vulkan for GPU render)
output_png = True# Output frame to PNG files (for later conversion to video), good for remote HPC
print("Output frames to GUI window{}, and PNG files{}".format(" enabled" if output_gui else "disabled", " enabled" if output_png else "disabled"))

# More bits = higher resolution, more accurate simulation, but slower and more memory usage
particle_quality_bits = 8#13 # Bits for particle count base unit, e.g. 13 = 2^13 = 8192 particles
grid_quality_bits = 7 # Bits for grid nodes base unit in a direction, e.g. 7 = 2^7 = 128 grid nodes in a direction
quality = 6 # Resolution multiplier that affects both particles and grid nodes by multiplying their base units w.r.t. dimensions

grid_length = 102.4  # Max length of the simulation domain in any direction [meters]
 
# Best to use powers of 2 for mem allocation, e.g. 0.5, 0.25, 0.125, etc. 
# Note: there are buffer grid-cells on either end of each dimension for multi-cell shape-function kernels and BCs
grid_ratio_x = 1.0 
grid_ratio_y = 0.125
grid_ratio_z = 0.125
grid_length_x = grid_length * ti.max(0.0, ti.min(1.0, grid_ratio_x))
grid_length_y = grid_length * ti.max(0.0, ti.min(1.0, grid_ratio_y))
grid_length_z = grid_length * ti.max(0.0, ti.min(1.0, grid_ratio_z))


n_grid_base = 2 ** grid_quality_bits # Using pow-2 grid-size for improved GPU mem usage / performance 
n_grid = n_grid_base * quality

n_grid = 2048
n_grid_x = int(ti.max(n_grid * ti.min(grid_ratio_x, 1), 1))
n_grid_y = int(ti.max(n_grid * ti.min(grid_ratio_y, 1), 1))
n_grid_z = int(ti.max(n_grid * ti.min(grid_ratio_z, 1), 1))
n_grid_total = int(ti.max(n_grid_x,1) * ti.max(n_grid_y,1))
dx, inv_dx = float(grid_length / n_grid), float(n_grid / grid_length)

n_particles_base = 2 ** particle_quality_bits # Better ways to do this, shouldnt have to set it manually
n_particles = n_particles_base * (quality**DIMENSIONS)
n_particles = 200000
downsampling = True
downsampling_ratio = 100 # Downsamples by 100x
# n_particles_water = (0.9 * 0.2 * grid_length * grid_length) * n_grid_base**2


print("Number of Particles: ", n_particles)
T.sleep(1)
print("Downsampling:{}".format(" enabled" if downsampling else "disabled"))
T.sleep(1)
print("Number of Downsampled Particles: ", int(n_particles / downsampling_ratio))
T.sleep(1)
print("Number of Grid-Nodes each Direction: ", n_grid_x, n_grid_y, n_grid_z)
print("dx: ", dx)

# Material properties
particles_per_dx = 4
particles_per_cell = particles_per_dx ** DIMENSIONS
print("Particles-per-Dx: ", particles_per_dx)
print("Particles-per-Cell: ", particles_per_cell)
particle_spacing_ratio = 1.0 / particles_per_dx
particle_spacing =  dx * particle_spacing_ratio
particle_volume_ratio = 1.0 / particles_per_cell
particle_volume = (dx ** DIMENSIONS) * particle_volume_ratio
p_vol, p_rho = particle_volume, 1000.0
p_mass = p_vol * p_rho
E, nu = 1e7, 0.2  # Young's modulus and Poisson's ratio
# TODO: Define material laws for various materials
gamma_water = 7.15 #Ratio of specific heats for water 
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
fps = 20
time_delta = 1.0 / fps


# Piston Physics
piston_amplitude = float(3.6) # 4 meters max range on piston's stroke in the OSU LWF
piston_period = float(180.0)
piston_pos = np.asarray(np.array([0.0, 0.0, 0.0])  \
            + (grid_length * np.array([0.0, 0.0, 0.0]))  \
            + (dx * np.array([4, 0, 0])), dtype=float) # Initial [X,Y,Z] position of the piston face
piston_start_x = 4 * dx / grid_length
piston_travel_x = piston_amplitude / grid_length
piston_wait_time = 1.0 # don't immediately start the piston, let things settle with gravity first

# Calc timestep based on elastic moduli of materials
CFL = 0.5 # CFL stability number. Typically 0.3 - 0.5 is good
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
board_states = ti.Vector.field(DIMENSIONS, float)
time = 0.0

#Define some parameters we would like to track
data_to_save = [] #used for saving positional data for particles 
v_data_to_save = []
bounds = [[0.0, 0.9], [0.0, 0.5]]
# bounds = [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]] # For 3D
vel_mean = []
vel_std = []
acc_mean = []
acc_std = []

# Material Ids
material_id_dict_gns = { "Water": 5, "Sand": 6, "Debris": 0, 
                    "Piston": 0, "Boundary": 3} # GNS Mapping Dict from Dr. Kumar

material_id_dict_mpm = { "Water": 0, "Snow": 1, "Debris": 2, "Sand": 3, "Steel": 4, 
                    "Piston": 5, "Boundary": 6} # Taichi/our Material Mapping
    

x = ti.Vector.field(DIMENSIONS, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(DIMENSIONS, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(DIMENSIONS, DIMENSIONS, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(DIMENSIONS, DIMENSIONS, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation

if DIMENSIONS == 2:
    grid_tuple = (n_grid_x, n_grid_y)
elif DIMENSIONS == 3:
    grid_tuple = (n_grid_x, n_grid_y, n_grid_z)

grid_v = ti.Vector.field(DIMENSIONS, dtype=float, shape=grid_tuple)  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=grid_tuple)  # grid node mass
gravity = ti.Vector.field(DIMENSIONS, dtype=float, shape=())

ti.root.place(board_states)

#Future class for different complex materials possibly??
#class material_models():
#    def __init__(self) -> None:


@ti.func
def update_material_properties(p):
    # Hardening coefficient: snow gets harder when compressed
    # Metals between h = .1-.5
    h = 1.0
    if material[p] == material_id_dict_mpm["Water"]: # 0
        h = 1.0
    if material[p] == material_id_dict_mpm["Snow"]: # 1 
    # Fixed-Corotated Hyper-elastic material: broad debris / jelly / plastic behavior
        h = 1.0 # Do not scale elastic moduli by default
    if material[p] == material_id_dict_mpm["Debris"]: # 2
        h = 1.0
    else:
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p])))) # Don't calc this unless used, expensive operation

    mu, la = mu_0 * h, lambda_0 * h # adjust elastic moduli based on hardening coefficient

    if material[p] == material_id_dict_mpm["Water"]: # 0:  # liquid 
        mu = 0.0 # assumed no shear modulus...

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
        if material[p] == material_id_dict_mpm["Debris"]: # 2:  # Snow-like material
            new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
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

@ti.kernel
def substep():
    clear_grid()
    p2g()
    update_grid()
    g2p()
    handle_piston_collisions()

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

        #N eo-hookean formulation for Cauchy stress from first Piola-Kirchoff stress
        stress = compute_stress(mu, la, F[p])

        # Reset deformation gradient to avoid numerical instability
        if material[p] == material_id_dict_mpm["Water"]: # 0
            F[p] = ti.Matrix.identity(float, DIMENSIONS) * ti.sqrt(J)
            
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
        if i < 3 and grid_v[i, j][0] < 0:
            grid_v[i, j][0] = 0  # Boundary conditions
        if i > n_grid_x - 3 and grid_v[i, j][0] > 0:
            grid_v[i, j][0] = 0
        if j < 3 and grid_v[i, j][1] < 0:
            grid_v[i, j][1] = 0
        if j > n_grid_y - 3 and grid_v[i, j][1] > 0:
            grid_v[i, j][1] = 0
    elif ti.static(DIMENSIONS == 3):
        if i < 3 and grid_v[i, j, k][0] < 0: 
            grid_v[i, j, k][0] = 0
        if i > n_grid_x - 3 and grid_v[i, j, k][0] > 0: 
            grid_v[i, j, k][0] = 0
        if j < 3 and grid_v[i, j, k][1] < 0: 
            grid_v[i, j, k][1] = 0
        if j > n_grid_y - 3 and grid_v[i, j, k][1] > 0: 
            grid_v[i, j, k][1] = 0
        if k < 3 and grid_v[i, j, k][2] < 0: 
            grid_v[i, j, k][2] = 0
        if k > n_grid_z - 3 and grid_v[i, j, k][2] > 0: 
            grid_v[i, j, k][2] = 0


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

@ti.func
def handle_piston_collisions():
    for p in x:
        # Apply piston force based on Hooke's law
        piston_pos_current = board_states[None][0]
        if x[p][0] < piston_pos_current:
            # Using separable contact, i.e. water doesn't stick if not being pushed
            displacement_into_piston = ti.max(piston_pos_current - x[p][0], 0.0)
            piston_spring_constant = p_mass / dt  # Assume a 1.0 kg mass 
            force = ti.max(piston_spring_constant * displacement_into_piston, 0.0)  # Hooke's law: F = k * x
            piston_escape_velocity = force / p_mass * dt  # v = F / m * dt
            piston_escape_velocity = ti.min(piston_escape_velocity, max_vel)  # Cap the velocity to prevent instability
            v[p][0] = ti.max(v[p][0], piston_escape_velocity)  # Stop the particle from moving into the piston

@ti.func
def erf_approx(x):
    """Needed an approximation to the gauss error function (math lib doesnt work with taichi)
    
    From: https://en.wikipedia.org/wiki/Error_function
    """
    # Approximation constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # Save the sign of x
    sign = 1
    if x < 0:
        sign = -1
    x = ti.abs(x)

    # Abramowitz and Stegun formula 
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * ti.exp(-x * x)

    return sign * y

@ti.kernel
def move_board_solitary():
    t = time
    b = board_states[None]
    b[1] += dt  # Adjusting for the coordinate frame

    if b[1] >= 2 * piston_period:
        b[1] = 0

    b[0] += piston_amplitude * ((erf_approx(t - 2.5 - 1e-2) + 1.0) / 2.0) #Soliton wave
    
    # Ensure the piston stays within the boundaries
    b[0] = ti.max(0.0, ti.min(b[0], piston_pos[0] + piston_amplitude))
    
    # Store the updated state back to the field
    board_states[None] = b

@ti.kernel
def reset():
    water_ratio_denominator = 64
    group_size = n_particles // water_ratio_denominator
    basin_row_size = int(ti.floor((1.0 - piston_start_x) * n_grid_x * particles_per_dx) - 3)
    debris_row_size = int(ti.floor(4 * particles_per_dx))
    for i in range(n_particles):
        row_size = basin_row_size
        # j = i // row_size
        water_ratio_numerator = water_ratio_denominator - 1
        n_water_particles = water_ratio_numerator * group_size
        if i < n_water_particles:
            # ppc = 4
            x[i] = [
                # ti.random() * 0.8 + 0.01 * (i // group_size),  # Fluid particles are spread over a wider x-range
                # ti.random() * 0.1 + 0.01 * (i // group_size)  # Fluid particles are spread over a wider y-range
                (piston_start_x * grid_length) + (dx * particle_spacing_ratio) * (i % row_size),  # Fluid particles are spread over a wider x-range
                (4 * dx) + (dx * particle_spacing_ratio) * (i // row_size)  # Fluid particles are spread over a wider y-range
            ]
            material[i] = 0  # fluid
        else:
            # Choose shape
            shape = 0
            if shape == 0:
                id = i % (n_water_particles)
                row_size = debris_row_size
                block_size = row_size**2
                debris_particle_x = ti.min(grid_length_x, (4*dx ) + (grid_length * (piston_start_x + piston_travel_x)) + (dx * particle_spacing_ratio) * ((id % row_size**2) % row_size) + grid_length * (16 * dx / grid_length) * (id // (row_size**2)))
                debris_particle_y = ti.min(grid_length_y, (4*dx) + (dx * (1 + particle_spacing_ratio * n_water_particles // basin_row_size)) + (dx * particle_spacing_ratio * ((id % row_size**2) // row_size)))

                x[i] = [
                    debris_particle_x,  # Block particles are confined to a smaller x-range
                    debris_particle_y   # Block particles are confined to a smaller y-range
                ]
            material[i] = 1  # Fixed-Corotated Hyper-elastic debris (e.g. for simple plastic, metal, rubber)
  
        v[i] = [0.0, 0.0]
        F[i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, DIMENSIONS, DIMENSIONS)
    board_states[None] = [float(piston_pos[0]), float(piston_pos[1])]  # Initial piston position

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
    vel = np.stack(v_data_to_save,axis=0)
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
    
    #Formatting enforced
    # Might want to replace 0.0025 with the dt actually used, etc., but maybe its like this for a reason
    metadata = {
        "bounds": bounds,
        "sequence_length": sequence_length, 
        "default_connectivity_radius": 0.5, 
        "dim": DIMENSIONS, 
        "dt": time_delta, 
        "dx": dx,
        "critical_time_step": dt,
        "vel_mean": vel_mean, #[5.123277536458455e-06, -0.0009965205918140803], 
        "vel_std": vel_std, #[0.0021978993231675805, 0.0026653552458701774], 
        "acc_mean": acc_mean, #[5.237611158734309e-07, 2.3633027988858656e-07], 
        "acc_std": acc_std, #[0.0002582944917306106, 0.00029554531667679154]
    }
    
    
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

        # Save to HDF5
        #with h5py.File(f'{cwd_path}/unspecified_sim_data.h5', 'w') as f:
        #    f.create_dataset('pos_data', data=downsampled_data)
        #    f.create_dataset('material_ids', data=downsampled_mat_data)
        
    print("Simulation Data Saved to: ", file_path)

@ti.kernel
def get_color(p: ti.i32) -> ti.i32:
    """assigns color from palette based on material
    args:
        p: particle index
    returns:
        an integer representing what color to use from the palette
    """
    if ti.static(material[p] == material_id_dict_mpm["Water"]):
        return 0  # index of water color in palette
    elif ti.static(material[p] == material_id_dict_mpm["Debris"]):
        return 1  # index of debris color in palette
    else:
        return 2  # default color index

# Define a Taichi field to store the result
#def downsample(X_data):

    # Reshape the array
#    Y = X_data.reshape(2, sequence_length, 2, 750, 2, 2)

    # Sum over the specified axes
#    Z = np.sum(Y, axis=(0, 2, 4)) / 8

    # Squeeze the resulting array to ensure removal of singleton dimensions
#    Z = np.squeeze(Z)

#    return Z

#Simulation Prerequisites 

data_designation = str(input('What is the output particle data for? Select: Rollout(R), Training(T), Valid(V) [Waiting for user input...] --> '))
# sequence_length = int(input('How many time steps to simulate? --> ')) 
fps = int(input('How many frames-per-second (FPS) to output? [Waiting for user input...] -->'))
sequence_length = int(input('How many seconds to run this simulations? [Waiting for user input...] --> ')) * fps # May want to provide an FPS input 

if DIMENSIONS == 2:
    gravity[None] = [0.0, -9.80665] # Gravity in m/s^2, this implies use of metric units
elif DIMENSIONS == 3:
    gravity[None] = [0.0, -9.80665, 0.0] # Gravity in m/s^2, this implies use of metric units

palette = [0x068587, 0xED553B, 0xEEEEF0,0x2E4057, 0xF0C987,0x6D214F]

gui_background_color_white = 0xFFFFFF # White or black generally preferred for papers / slideshows, but its up to you
gui_background_color_taichi= 0x112F41 # Taichi default background color, may be easier on the eyes

print("\nPress R to reset.")
gui_res = min(1080, n_grid) # Set the resolution of the GUI
gui = ti.GUI("Digital Twin of the NSF OSU LWF Facility - Tsunami Debris Simulation in the Material Point Method", res=gui_res, background_color=gui_background_color_white)
reset()

# Saving Figures of the simulation
base_frame_dir = './Flume/figures/'
os.makedirs(base_frame_dir, exist_ok=True) # Ensure the directory exists
frame_paths = []


for frame in range(sequence_length):  
    if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == "r":
                print("Resetting...")
                reset()
                data_to_save = []
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break

    # for s in range(int(2e-3 // dt)): # Will need to double-check the use of 2e-3, dt, etc.
    for s in range(int((1.0/fps) // dt)): 
        substep()
        if time >= piston_wait_time:  # Wait to activate piston until particles settle
            move_board_solitary()
        time += dt # Update time by dt so that the time used in move_board_solitary() is accurate, otherwise the piston moves only once every frame position-wise which causes instabilities


    print(f't = {round(time,3)}')

    
    #Change to tiachi fields probably
    data_to_save.append(x.to_numpy())
    v_data_to_save.append(v.to_numpy())
    
    clipped_material = np.clip(material.to_numpy(), 0, len(palette) - 1) #handles error where the number of materials is greater len(palette)
    gui.circles(
        x.to_numpy() / grid_length,
        radius=1.0,
        palette=palette,
        palette_indices=clipped_material,
    )
    #gui.circles(
    #x.to_numpy() / grid_length,
    #radius=1.5,
    #palette=palette,
    #palette_indices=get_color(np.arange(n_particles))
    #)

    # Render the moving piston
    piston_pos_current = board_states[None][0]
    piston_draw = np.array([board_states[None][0] / grid_length, board_states[None][1] / grid_length])
    
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
    #print(f"Time: {time}, Number of particles: {n_particles}, Board position: {board_states[None]}")


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
#save_simulation()

#if using save_sim.py script
ss.save_sim_data(data_designation, data_to_save, v_data_to_save, material, 
                 bounds, sequence_length, DIMENSIONS, time_delta, dx, dt)