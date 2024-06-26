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

DIMENSIONS = 2 # DIMENSIONS, 2D or 3D
output_gui = True # Output to GUI window (original, not GGUI which requires vulkan for GPU render)
output_png = False# Output frame to PNG files (for later conversion to video), good for remote HPC
print("Output frames to GUI window{}, and PNG files{}".format(" enabled" if output_gui else "disabled", " enabled" if output_png else "disabled"))


ti.init(arch=ti.gpu)  # Try to run on GPU

# More bits = higher resolution, more accurate simulation, but slower and more memory usage
particle_quality_bits = 13 # Bits for particle count base unit, e.g. 13 = 2^13 = 8192 particles
grid_quality_bits = 7 # Bits for grid nodes base unit in a direction, e.g. 7 = 2^7 = 128 grid nodes in a direction
quality = 6 # Resolution multiplier that affects both particles and grid nodes by multiplying their base units w.r.t. dimensions

grid_length = 102.4  # Max length of the simulation domain in any direction [meters]
 
# Best to use powers of 2 for mem allocation, e.g. 0.5, 0.25, 0.125, etc. 
# Note: there are buffer grid-cells on either end of each dimension for multi-cell shape-function kernels and BCs
grid_ratio_x = 1.0 
grid_ratio_y = 0.125
grid_ratio_z = 0.0
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
n_particles = 1000000
# n_particles_water = (0.9 * 0.2 * grid_length * grid_length) * n_grid_base**2


print("Number of Particles: ", n_particles)
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

piston_amplitude = float(3.6) # 4 meters max range on piston's stroke in the OSU LWF
piston_period = float(180.0)
piston_pos = np.asarray(np.array([0.0, 0.0, 0.0])  \
            + (grid_length * np.array([0.0, 0.0, 0.0]))  \
            + (dx * np.array([4, 0, 0])), dtype=float) # Initial [X,Y,Z] position of the piston face
piston_start_x = 4 * dx / grid_length
piston_travel_x = piston_amplitude / grid_length

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
bounds = [[0.1, 0.9], [0.1, 0.9]]
# bounds = [[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]] # For 3D
vel_mean = []
vel_std = []
acc_mean = []
acc_std = []


x = ti.Vector.field(DIMENSIONS, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(DIMENSIONS, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(DIMENSIONS, DIMENSIONS, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(DIMENSIONS, DIMENSIONS, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_tuple = (n_grid_x, n_grid_y) #if DIMENSIONS == 2 else  (n_grid, n_grid, n_grid)
grid_v = ti.Vector.field(DIMENSIONS, dtype=float, shape=grid_tuple)  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=grid_tuple)  # grid node mass
gravity = ti.Vector.field(DIMENSIONS, dtype=float, shape=())
#attractor_strength = ti.field(dtype=float, shape=())
#attractor_pos = ti.Vector.field(DIMENSIONS, dtype=float, shape=())
#pressure = ti.field(dtype=ti.f32, shape=n_particles)  # Pressure field
pressure = 0
ti.root.place(board_states)

@ti.func
def update_material_properties(p):
    # Hardening coefficient: snow gets harder when compressed
    h = 1.0
    if material[p] == 0: 
        h = 1.0
    if material[p] == 1:  # Fixed-Corotated Hyper-elastic material: broad debris / jelly / plastic behavior
        h = 1.0 # Do not scale elastic moduli by default
    if material[p] == 2:
        h = 1.0
    else:
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p])))) # Don't calc this unless used, expensive operation

    mu, la = mu_0 * h, lambda_0 * h # adjust elastic moduli based on hardening coefficient
    if material[p] == 0:  # liquid 
        mu = 0.0 # assumed no shear modulus...

    return h, mu, la

@ti.kernel
def substep():
    # if DIMENSIONS == 2:
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    # elif DIMENSIONS == 3:
    #     for i, j, k in grid_m:
    #         grid_v[i, j, k] = [0, 0, 0]
    #         grid_m[i, j, k] = 0
    # else:
        # ti.static_print("Error: Invalid number of DIMENSIONS")
        
    # We will need to detangle these material laws into their own functions / objects
    # Taichi did things improperly here 
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2] or Weights for MPM
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, DIMENSIONS) + dt * C[p]) @ F[p]
        
        # Hardening coefficient and Lame parameter updates
        h, mu, la = update_material_properties(p)
            
        #U, sig, V = ti.svd(F[p]) # Singular Value Decomposition of deformation gradient (on particle)

        J = ti.math.determinant(F[p])  #particle volume ratio = V /Vo
        # J=1 undeformed material; J<1 compressed material; J>1 expanded material
        
        # may need to reformulate since we want as close to water incompressability as possible
        # Useful paper https://www.sciencedirect.com/science/article/pii/S2590055219300319
        pressure = bulk_modulus*( (J**-gamma_water) - 1 ) #Tait formulation for weakly compressible fluid (isentropic)
        # pressure = bulk_modulus*( (1/J) - 1 ) #Pressure for solid 
        #J = 1.0
        #for d in ti.static(range(DIMENSIONS)):
        #    new_sig = sig[d, d]
        #    if material[p] == 2:  # Snow-like material
        #        new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
        #    Jp[p] *= sig[d, d] / new_sig # stable?
        #    sig[d, d] = new_sig
        #    J *= new_sig
        if material[p] == 0: # water
            # Reset deformation gradient to avoid numerical instability
            # if DIMENSIONS == 2:
            F[p] = ti.Matrix.identity(float, DIMENSIONS) * ti.sqrt(J)
            # elif DIMENSIONS == 3:
            #     F[p] = ti.Matrix.identity(float, DIMENSIONS) * ti.cbrt(J)
            # else:
                # ti.static_print("Error: Invalid number of DIMENSIONS")
            
       # elif material[p] == 2:
            # Reconstruct elastic deformation gradient after plasticity
       #     F[p] = U @ sig @ V.transpose() # Singular value decomposition, good for large deformations on fixed-corotated model 
        
        #stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, DIMENSIONS) * la * J * (
        #    J - 1
        #)

        # Dp_inv = 3 * inv_dx * inv_dx # Applies only to BSpline Cubic kernel in APIC/MLS-MPM / maybe PolyPIC
        # Dp_inv = 4 * inv_dx * inv_dx # Applies only to BSpline Quadratic kernel in APIC/MLS-MPM / maybe PolyPIC
        #stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        #affine = stress + p_mass * C[p]
        #affine = pressure + p_mass * C[p] #Using pressure instead
        affine = pressure * F[p].inverse().transpose() + p_mass * C[p]
        #print(stress)

        for i, j in ti.static(ti.ndrange(3, 3)):
            # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            # Momentum to velocity
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j] += dt * gravity[None]   # gravity
            # dist = attractor_pos[None] - dx * ti.Vector([i, j])
            # grid_v[i, j] += dist / (grid_length * 0.01 + dist.norm()) * attractor_strength[None] * dt 
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid_x - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid_y - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0

    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, DIMENSIONS)
        new_C = ti.Matrix.zero(float, DIMENSIONS, DIMENSIONS)
        for i, j in ti.static(ti.ndrange(3, 3)):
            # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection

        # if x[p][1] > grid_length_y:  # Upper boundary
        #     x[p][1] = grid_length_y
        #     if v[p][1] > 0:
        #         v[p][1] = 0  # Stop upward velocity
                
    # Piston Collisions
    for p in x:
        # if x[p][0] < piston_pos[0]:  # Adjust the threshold as needed
        #     v[p][0] = ti.max(v[p][0], ti.max(1.0 * ti.max(piston_pos[0] - x[p][0], 0.0) / dt  - v[p][0], 1.0 * ti.max(piston_pos[0] - x[p][0], 0.0) / dt * 1.0) )   # Adjust the force strength as needed
        #     displacement_into_piston = ti.max(piston_pos_current - x[p][0], 0.0)
        #     piston_escape_velocity = 
        #     v[p][0] = ti.max(v[p][0], piston_escape_velocity)  # Stop the particle from moving into the piston
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
            
            # v[p][0] += ti.max(force * dt - v[p][0], force * dt * 1.0)  # Apply the force to the velocity

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
    
# @ti.kernel
# def move_board():
#     b = board_states[None]
#     b[1] += .2 #adjusting for the coordinate frame
#     b[1] += dt
#     period = 180
#     vel_strength = 0.6 # Make this analytical w.r.t. time - position
#     if b[1] >= 2 * period:
#         b[1] = 0
#     # Update the piston position
#     piston_motion_scale = 0.01 # Assume a 100 meter flume length for scaling
#     b[0] += -(ti.sin(b[1] * np.pi / period) * vel_strength) * time_delta * piston_motion_scale
#     # Ensure the piston stays within the boundaries
#     b[0] = ti.max(0, ti.min(b[0], 0.12))
#     #b[0] = ti.max(0.88, ti.min(b[0], 1.0))  # boundaries for the right side if we want piston there
#     board_states[None] = b

@ti.kernel
def move_board_solitary():
    wait_time = 5.0 # don't immediately start the piston, let things settle with gravity first
    t = time - wait_time if time - wait_time > 0.0 else 0.0
    b = board_states[None]
    b[1] += dt  # Adjusting for the coordinate frame
    # b[1] += 0.2

    #vel_strength = 2.0

    if b[1] >= 2 * piston_period:
        b[1] = 0

    # piston_motion_scale = 0.01125 # Assume a ~88.89 meter flume length for scaling
    # piston_amplitude = float(piston_amplitude * 1.0) # Double amplitude for more interesting visuals, remove later
    # Update the piston position using the error function approximation function
    # b[0] += (piston_pos[0]) + piston_amplitude * ((erf_approx(t - 2.5 - 1e-2) + 1.0) / 2.0) #Soliton wave
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
        if i < water_ratio_numerator * group_size:
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
            shape = 2
            if shape == 0:
                # Initialize debris particles from circles for jelly material
                angle = 2 * np.pi * ti.random()
                radius = 0.005 * ti.random()
                x[i] = [
                    ti.random() * 0.1 + 0.1 * (i // group_size) + radius * ti.cos(angle),  # Circle particles in smaller x-range
                    ti.random() * .9 + 0.1 * (i // group_size) + radius * ti.sin(angle)  # Circle particles in smaller y-range

                ]
            elif shape == 1:
                # Initialize circles for jelly material
                angle = 2 * np.pi * ti.random()
                radius = 0.05 * ti.random()
                x[i] = [
                    ti.random() * 0.05 + 0.2 * (i // group_size) + radius * ti.cos(angle),  # Circle particles in smaller x-range
                    ti.random() * 0.05 + 0.05 * (i // group_size) + radius * ti.sin(angle)  # Circle particles in smaller y-range
                ]
            elif shape == 2:
                id = i % (water_ratio_numerator * group_size)
                row_size = debris_row_size
                block_size = row_size**2
                debris_particle_x = ti.min(grid_length_x, (4*dx ) + (grid_length * (piston_start_x + piston_travel_x)) + (dx * particle_spacing_ratio) * ((id % row_size**2) % row_size) + grid_length * (16 * dx / grid_length) * (id // (row_size**2)))
                debris_particle_y = ti.min(grid_length_y, (4*dx) + (dx * (1 + particle_spacing_ratio * water_ratio_numerator * group_size // basin_row_size)) + (dx * particle_spacing_ratio * ((id % row_size**2) // row_size)))

                x[i] = [
                    debris_particle_x,  # Block particles are confined to a smaller x-range
                    debris_particle_y   # Block particles are confined to a smaller y-range
                ]
            material[i] = 1  # Fixed-Corotated Hyper-elastic debris (e.g. for simple plastic, metal, rubber)
        #x[i] = [
        #    ti.random() * 0.2 + 0.1 + 0.4 * (i // group_size),
        #    ti.random() * 0.2 + 0.02 + 0.02 * (i // group_size),
        #]
        #material[i] = min(i // group_size, 1)  # 0: fluid 1: jelly 2: snow
        v[i] = [0.0, 0.0]
        F[i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        Jp[i] = 1.0
        C[i] = ti.Matrix.zero(float, DIMENSIONS, DIMENSIONS)
    board_states[None] = [float(piston_pos[0]), float(piston_pos[1])]  # Initial piston position

def save_metadata(file_path):
    """Save metadata.json to file
    Args:
        None
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
    global data_to_save

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
    
    material_id_dict_gns = { "Water": 5, "Sand": 6, "Debris": 0, "Piston": 0, "Boundary": 3} # GNS Mapping Dict from Dr. Kumar
    material_id_dict_mpm = { "Water": 0, "Snow": 1, "Debris": 2, "Sand": 3, "Piston": 4, "Boundary": 5}

    material_numpy = material.to_numpy()
    mat_data_tmp = np.where(material_numpy == material_id_dict_mpm["Water"], material_id_dict_gns["Water"] + (0 * material_numpy), material_numpy)
    #mat_data = np.where(material_numpy == 0, material_id_dict['Water'], material_numpy)

    mat_data = np.asarray(mat_data_tmp, dtype=object)
    pos_data = np.stack(data_to_save, axis=0)

    # Perform downsampling for GNS    
    downsampled_mat_data = mat_data[::100]
    downsampled_data = pos_data[:,::100,:]


    #check version of numpy >= 1.22.0
    # Newer versions of numpy require the dtype to be explicitly set to object, I think, for some python versions
    # Should add a check for the python version as well

    if (np.version.version >= '1.22.0'):
        print("Using numpy version (>= 1.22.0), may require alternative approach to save npz files (e.g. dtype=object): ", np.version.version)
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
        np.savez_compressed(f'{file_path}/valid.npz', **simulation_data)
        
    else:
        output_file_path = os.path.join(cwd_path, "unspecified_sim_data.npz")
        np.savez_compressed("unspecified_sim_data.npz", **simulation_data)
        
    print("Simulation Data Saved to: ", file_path)
    
# Define a Taichi field to store the result
def downsample(X_data):

    # Reshape the array
    Y = X_data.reshape(2, sequence_length, 2, 750, 2, 2)

    # Sum over the specified axes
    Z = np.sum(Y, axis=(0, 2, 4)) / 8

    # Squeeze the resulting array to ensure removal of singleton dimensions
    Z = np.squeeze(Z)

    return Z

#Simulation Prerequisites 

data_designation = str(input('What is the output particle data for? Select: Rollout(R), Training(T), Valid(V) [Waiting for user input...] --> '))
# sequence_length = int(input('How many time steps to simulate? --> ')) 
fps = int(input('How many frames-per-second (FPS) to output? [Waiting for user input...] -->'))
sequence_length = int(input('How many seconds to run this simulations? [Waiting for user input...] --> ')) * fps # May want to provide an FPS input 
gravity[None] = [0.0, -9.80665] # Gravity in m/s^2, this implies use of metric units

palette = [0x068587, 0xED553B, 0xEEEEF0,0x2E4057, 0xF0C987,0x6D214F]

gui_background_color_white = 0xFFFFFF # White or black generally preferred for papers / slideshows, but its up to you
gui_background_color_taichi= 0x112F41 # Taichi default background color, may be easier on the eyes

print("\nPress R to reset.")
gui_res = min(1080, n_grid) # Set the resolution of the GUI
gui = ti.GUI("Digital Twin of the NSF OSU LWF Facility - Tsunami Debris Simulation in the Material Point Method", res=gui_res, background_color=gui_background_color_white)
reset()

for frame in range(sequence_length):  
    if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == "r":
                print("Resetting...")
                reset()
                data_to_save = []
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break

    # for s in range(int(2e-3 // dt)): # Will need to double-check the use of 2e-3, dt, etc.
    for s in range(int((1.0/fps) // dt)): # Will need to double-check the use of 2e-3, dt, etc
        substep()

        move_board_solitary()
        time += dt # Update time by dt so that the time used in move_board_solitary() is accurate, otherwise the piston moves only once every frame position-wise which causes instabilities
        
    # time += time_delta / 100
    # time = frame * time_delta
    print(f't = {round(time,3)}')

    
    #Change to tiachi fields probably
    data_to_save.append(x.to_numpy())
    v_data_to_save.append(v.to_numpy())

    # time += time_delta
    print(f't = {round(time,3)}')
    
    clipped_material = np.clip(material.to_numpy(), 0, len(palette) - 1) #handles error where the number of materials is greater len(palette)
    gui.circles(
        x.to_numpy() / grid_length,
        radius=1.0,
        palette=palette,
        palette_indices=clipped_material,
    )
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


    # print(f'Frame {i} is recorded in {frame_filename}')
    # gui.show(filename)  # export and show in GUI

    frame_filename = f'dataset/figures/frame_{frame:05d}.png'   # create filename with suffix png
    if output_png and output_gui:
        # gui.show("./dataset/figures/" + frame_filename)
        gui.show(frame_filename)

    elif output_png and output_gui == False:
        tools.imwrite(x.to_numpy(), f'./dataset/figures/{frame_filename}')
    elif output_gui and output_png == False:
        gui.show()
    else:
        print("WARNING - No output method selected, frame not saved or displayed...")
    
    if output_gui == False:
        continue

#Prep for GNS input
save_simulation()






