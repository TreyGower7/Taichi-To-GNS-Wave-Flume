import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# Define simulation parameters
nx, ny, nz = 128, 64, 64  # Grid resolution
n_particles = 300000  # Number of particles
dx = 1.0 / nx  # Grid cell size
dt = 1e-4  # Time step size

# Particle data
x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
m = ti.field(dtype=ti.f32, shape=n_particles)

# Grid data
grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
grid_m = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

@ti.kernel
def initialize():
    for i in range(n_particles):
        x[i] = ti.Vector([ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.05, ti.random() * 0.4 + 0.3])
        v[i] = ti.Vector([0.0, 0.0, 0.0])
        m[i] = 1.0

@ti.kernel
def reset_grid():
    for i, j, k in grid_m:
        grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
        grid_m[i, j, k] = 0.0

@ti.kernel
def p2g():
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            grid_v[base + offset] += weight * m[p] * v[p]
            grid_m[base + offset] += weight * m[p]

@ti.kernel
def grid_op():
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] /= grid_m[i, j, k]
        grid_v[i, j, k].y -= 9.8 * dt  # Gravity
        
        # Simple boundary conditions
        if i < 2 or i > nx - 3 or j < 2 or k < 2 or k > nz - 3:
            grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def g2p():
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0, 0.0])
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i].x * w[j].y * w[k].z
            new_v += weight * grid_v[base + offset]
        v[p] = new_v
        x[p] += dt * v[p]

# Initialize particles
initialize()

# Create window
window = ti.ui.Window("3D MPM Wave Flume", (1024, 768), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# Set initial camera position
camera.position(1.5, 1.5, 1.5)
camera.lookat(0.5, 0.3, 0.5)

# Main simulation loop
while window.running:
    reset_grid()
    p2g()
    grid_op()
    g2p()
    
    # Set the camera
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    # Set the light
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))

    # Draw the particles
    scene.particles(x, radius=0.005, color=(0, 0.5, 0.5))

    # Render the scene
    canvas.scene(scene)
    window.show()