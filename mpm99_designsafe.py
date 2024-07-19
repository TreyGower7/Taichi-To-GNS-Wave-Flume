import taichi as ti
from taichi import tools
import numpy as np
import platform # For getting the operating system name, taichi may already have something for this
import os
import json
import math
import imageio
import time as T
import ffmpeg
from matplotlib import cm

ti.init(arch=ti.gpu)  # Try to run on GPU
quality = 1  # Use a larger value for higher-res simulations
n_particles_direction = 100
dim = 2
n_particles, n_grid = (n_particles_direction ** dim) * quality**dim, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 2e-5 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 5e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass


rgba = ti.Vector.field(4, dtype=ti.f32, shape=(n_particles_direction, n_particles_direction ))

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
        # F[p]: deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # h: Hardening coefficient: snow gets harder when compressed
        h = ti.exp(10 * (1.0 - Jp[p]))
        if material[p] == 2:  # Snow
            h *= 0.1
        if material[p] == 1:  # jelly, make it softer
            h = 1.0
            
        mu, la = mu_0 * h, lambda_0 * h
        
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        # Avoid zero eigenvalues because of numerical errors
        for d in ti.static(range(2)):
            sig[d, d] = ti.max(sig[d, d], 1e-6)
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
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
        # Loop over 3x3 grid node neighborhood
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]  # Momentum to velocity
            grid_v[i, j][1] -= dt * 50  # gravity
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


group_size = n_particles // 3


@ti.kernel
def initialize():
    for i in range(n_particles):
        x[i] = [
            ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
            ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size),
        ]
        material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
        v[i] = ti.Matrix([0, 0])
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1

@ti.kernel
def fill_rgba():
    for i, j in rgba:
        rgba[i, j] = ti.Vector(
            [ti.random(), ti.random(), ti.random(), ti.random()])



pixels = ti.field(ti.u8, shape=(512, 512, 3))

@ti.kernel
def paint():
    for i, j, k in pixels:
        pixels[i, j, k] = ti.random() * 255


result_dir = "./results"
#video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=24, automatic_build=False)
#
#for i in range(50):
#    paint()
#
#    pixels_img = pixels.to_numpy()
#    video_manager.write_frame(pixels_img)
#    print(f'\rFrame {i+1}/50 is recorded', end='')

print()
#print('Exporting .mp4 and .gif videos...')
#video_manager.make_video(gif=True, mp4=True)
#print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
#print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')

def main():
    initialize()
    #gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    #while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
    #for s in range(int(2e-3 // dt)):
    #    substep()
        #gui.circles(
        #    x.to_numpy(),
        #    radius=1.5,
        #    palette=[0x068587, 0xED553B, 0xEEEEF0],
        #    palette_indices=material,
        #)
        # Change to gui.show(f'{frame:06d}.png') to write images to disk
        #gui.show()

    series_prefix = "mpm99_particles.ply"
    max_frames = 360
    for frame in range(max_frames):
        im = np.ones((4 * n_grid, 4 * n_grid, 3),dtype=np.uint8)*(0x112F41)
        
        steps_per_frame = int(1e-2 // dt)
        for s in range(steps_per_frame):
            substep()
            
        print("Frame: ", frame, " / ", max_frames)

        fill_rgba()
        # now adding each channel only supports passing individual np.array
        # so converting into np.ndarray, reshape
        # remember to use a temp var to store so you dont have to convert back
        np_pos = np.reshape(x.to_numpy(), (n_particles, 2))
        np_rgba = np.reshape(rgba.to_numpy(), (n_particles, 4))
        np_pixels = np.reshape(pixels.to_numpy(), (512, 512, 3))


        palette=[[0,0,125], [125,0,0],[240,240,240],[125,0,125],[125,125,0],[125,125,125]] # red, green, blue, cyan, yellow, white

        for p in range(np_pos.shape[0]):
            xc = np_pos[p][0]
            yc = np_pos[p][1]
            buffer_on_side = 32
            grid_length = float( (n_grid + 1) / (n_grid * 0.9))
            im[min(4*n_grid-1,math.floor(xc * n_grid * 4 / grid_length)), min(4*n_grid-1,math.floor(yc * n_grid * 4 / grid_length)), :] = np.array(palette[material[p]],dtype=np.uint8)[:]


        np_im = np.array(im, dtype=np.uint8)

        # create a PLYWriter
        writer = ti.tools.PLYWriter(num_vertices=n_particles)
        writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np.zeros((np_pos.shape[0],)))
        writer.add_vertex_rgba(
        np_rgba[:, 0], np_rgba[:, 1], np_rgba[:, 2], np_rgba[:, 3])
        writer.export_frame_ascii(frame, series_prefix)
        ti.tools.imwrite(np_im, f'frame_{frame:06d}.png')

if __name__ == "__main__":
    main()