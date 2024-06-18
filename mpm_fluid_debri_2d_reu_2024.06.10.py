import taichi as ti
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import json

ti.init(arch=ti.gpu)  # Try to run on GPU

quality = 2  # Use a larger value for higher-res simulations
n_particles, n_grid = 6000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 5e3, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
time_delta = 1.0 / 20.0

#Added parameters for piston and particle interaction
boundary_color = 0xEBACA2
board_states = ti.Vector.field(2, float)

#Define some parameters we would like to track
data_to_save = [] #used for saving positional data for particles 
v_data_to_save = []
bounds = [[0.1, 0.9], [0.1, 0.9]]
vel_mean = []
vel_std = []
acc_mean = []
acc_std = []


x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())
ti.root.place(board_states)

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Hardening coefficient: snow gets harder when compressed
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
        if material[p] == 1:  # jelly, make it softer
            h = .5
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0:
            # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 2:
            # Reconstruct elastic deformation gradient after plasticity
            F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (
            J - 1
        )
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
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
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            dist = attractor_pos[None] - dx * ti.Vector([i, j])
            grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection
        
     # Apply boundary collisions
     #   if x[p][1] < 0.1:  # Lower boundary
     #       x[p][1] = 0.1
     ##       if v[p][1] < 0:
     #           v[p][1] = 0  # Stop downward velocity

        if x[p][1] > 0.2:  # Upper boundary
            x[p][1] = 0.2
            if v[p][1] > 0:
                v[p][1] = 0  # Stop upward velocity
                
    # Piston Collisions
    for p in x:
        if x[p][0] < 0.05:  # Adjust the threshold as needed
            v[p][0] += 1 * dt  # Adjust the force strength as needed
        # Apply piston force based on Hooke's law
        piston_pos = board_states[None][0]
        if x[p][0] < piston_pos:
            displacement = piston_pos - x[p][0]
            force = 1 * displacement  # Hooke's law: F = k * x
            v[p][0] += force * dt / p_mass  # Apply the force to the velocity
        
@ti.kernel
def move_board():
    b = board_states[None]
    b[1] += .2 #adjusting for the coordinate frame
    period = 180
    vel_strength = 2.0
    if b[1] >= 2 * period:
        b[1] = 0
    # Update the piston position
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    # Ensure the piston stays within the boundaries
    b[0] = ti.max(0, ti.min(b[0], 0.12))
    #b[0] = ti.max(0.88, ti.min(b[0], 1.0))  # boundaries for the right side if we want piston there
    board_states[None] = b
    
@ti.kernel
def reset():
    group_size = n_particles // 3
    for i in range(n_particles):
        if i < group_size:
            x[i] = [
                ti.random() * 0.4 + 0.01 * (i // group_size),  # Fluid particles are spread over a wider x-range
                ti.random() * 0.3 + 0.01 * (i // group_size)  # Fluid particles are spread over a wider y-range
            ]
            material[i] = 0  # fluid
        else:
            # Choose shape
            shape = 0
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
                x[i] = [
                    ti.random() * 0.1 + 0.2 * (i // group_size),  # Block particles are confined to a smaller x-range
                    ti.random() * 0.1 + 0.05 * (i // group_size)     # Block particles are confined to a smaller y-range
                ]
            material[i] = 1  # jelly
        #x[i] = [
        #    ti.random() * 0.2 + 0.1 + 0.4 * (i // group_size),
        #    ti.random() * 0.2 + 0.02 + 0.02 * (i // group_size),
        #]
        #material[i] = min(i // group_size, 1)  # 0: fluid 1: jelly 2: snow
        v[i] = [1, 1]
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1
        C[i] = ti.Matrix.zero(float, 2, 2)
    board_states[None] = [0.02, 0]  # Initial piston position

def save_metadata():
    """Save metadata.json to file
    Args:
        None
    Returns:
        None
    """
    file_path = "./dataset"
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
    metadata = {
        "bounds": bounds,
        "sequence_length": sequence_length, 
        "default_connectivity_radius": 0.015, 
        "dim": 2, 
        "dt": 0.0025, 
        "vel_mean": vel_mean, #[5.123277536458455e-06, -0.0009965205918140803], 
        "vel_std": vel_std, #[0.0021978993231675805, 0.0026653552458701774], 
        "acc_mean": acc_mean, #[5.237611158734309e-07, 2.3633027988858656e-07], 
        "acc_std": acc_std, #[0.0002582944917306106, 0.00029554531667679154]
    }
    
        
      # Ensure the target directory exists
    os.makedirs(file_path, exist_ok=True)
    
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
    file_path = "/Users/treygower/code-REU/Physics-Informed-ML/dataset/"
    
    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # Stack the data along a new axis for formatting
    pos_data = np.stack(data_to_save, axis=0)
    
    #replacing material data with Dr. Kumars material ids
    material_data = np.where(material.to_numpy() == 0, 5, material.to_numpy())
    # Combine arrays into a single dictionary (Using a list does not work)
    combined_data = {
        'simulation_0':
        (
            pos_data,
            material_data
        )
    }

    if data_designation.lower() in ("r", "rollout"):
        np.savez_compressed(f'{file_path}/test.npz', simulation_trajectory=combined_data)

    elif data_designation.lower() in ("t", "train"):
        np.savez_compressed(f'{file_path}/train.npz', **combined_data)
        
    elif data_designation.lower() in ("v", "valid"):
        np.savez_compressed(f'{file_path}/valid.npz', simulation_trajectory=combined_data)
        
    else:
        np.savez_compressed("unspecified_sim_data.npz", simulation_trajectory=combined_data)
        
    print("Simulation Data Saved!\n")

#Simulation Prerequisites 
data_designation = str(input('Simulation Purpose: Rollout(R), Train(T), Valid(V) --> '))
sequence_length = int(input('How many time steps to simulate? --> '))
gravity[None] = [0, -9.81]
palette = [0x068587, 0xED553B, 0xEEEEF0,0x2E4057, 0xF0C987,0x6D214F]

print("\nPress R to reset.")
gui = ti.GUI("Taichi MPM-With-Piston", res=512, background_color=0x112F41)
reset()

for frame in range(sequence_length):  
    if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == "r":
                print("Resetting...")
                reset()
                data_to_save = []
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break

    for s in range(int(2e-3 // dt)):
        substep()
        move_board()
    
    # Export positions to numpy array
    data_to_save.append(x.to_numpy())
    v_data_to_save.append(v.to_numpy())
    
    clipped_material = np.clip(material.to_numpy(), 0, len(palette) - 1) #handles error where the number of materials is greater len(palette)
    gui.circles(
        x.to_numpy(),
        radius=1.5,
        palette=palette,
        palette_indices=clipped_material,
    )
    # Render the moving piston
    piston_pos = board_states[None][0]
    
    #print(piston_pos)
    gui.line(
        [piston_pos, 0], [piston_pos, 1],
        color=boundary_color,
        radius=2
    )
    gui.line(
        [0, .2], [1, .2],
        color=boundary_color,
        radius=2
    )


    gui.show()

#Prep for GNS input
save_simulation()
if data_designation.lower() in ("t", "train"):
    save_metadata()





