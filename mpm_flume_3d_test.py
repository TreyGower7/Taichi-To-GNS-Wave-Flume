import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

# Simulation parameters
dim = 2
n_particles = 16384
n_grid_x = 512
n_grid_y = 128
dx = 1.0 / n_grid_y
inv_dx = 1.0 / dx
dt = 1e-4
p_mass = 1.0
p_vol = 1.0
E = 400

# Particle and grid quantities
x = ti.Vector.field(dim, dtype=float, shape=n_particles)
v = ti.Vector.field(dim, dtype=float, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
J = ti.field(dtype=float, shape=n_particles)

grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid_x, n_grid_y))
grid_m = ti.field(dtype=float, shape=(n_grid_x, n_grid_y))

# Soliton wave parameters
A = 0.2  # Amplitude
c = 1.0  # Wave speed
x0 = -2.0  # Initial position

@ti.func
def soliton(x, t):
    return A / (math.cosh(0.5 * math.sqrt(3 * A) * (x - x0 - c * t)) ** 2)

@ti.kernel
def initialize():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.8 * n_grid_x * dx, ti.random() * 0.4 * n_grid_y * dx]
        v[i] = [0, 0]
        J[i] = 1

@ti.kernel
def apply_soliton(t: float):
    for i in range(n_particles):
        if x[i][0] < 10 * dx:
            x[i][1] += soliton(x[i][0], t) - soliton(x[i][0], t - dt)
            v[i][1] = (soliton(x[i][0], t) - soliton(x[i][0], t - dt)) / dt

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * p_vol * (J[p] - 1) * 4 * E * inv_dx * inv_dx
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j].y -= dt * 9.8  # gravity
        if j < 3 and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0  # Bottom boundary condition
        if j > n_grid_y - 3 and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0  # Top boundary condition
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx * inv_dx
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C

initialize()
gui = ti.GUI("MPM Wave Flume", (1024, 256))

for frame in range(10000):
    for s in range(50):
        apply_soliton(frame * 50 * dt + s * dt)
        substep()
    
    gui.clear(0x112F41)
    particles_numpy = x.to_numpy()
    gui.circles(particles_numpy / [n_grid_x * dx, n_grid_y * dx], radius=1.5, color=0x068587)
    gui.show()