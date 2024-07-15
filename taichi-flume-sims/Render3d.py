import taichi as ti
import mpm_fluid_debris_reu_2024_new as mpm_sim
import numpy as np

DIMENSIONS, grid_length, flume_width_3d, flume_height_3d, x, scene, camera, n_particles, canvas, gui, gui_res, material, palette = mpm_sim.send_render_data()

front_back_color = (166/255, 126/255, 71/255)  # Rust color in RGB
side_color = (197/255, 198/255, 199/255)  # Light grey color in RGB
bottom_color = (79/255, 78/255, 78/255)
background_color = (237/255, 235/255, 235/255)
flume_vertices = ti.Vector.field(3, dtype=float, shape=8)
bottom = ti.field(dtype=int, shape=6)  # Change to a 1D array
# Only way to render flume with differing colors
backwall = ti.field(dtype=int, shape=6)
frontwall = ti.field(dtype=int, shape=6)
sidewalls = ti.field(dtype=int, shape=6)
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

@ti.kernel
def copy_to_field(source: ti.types.ndarray(), target: ti.template()):
    for i in range(source.shape[0]):
        for j in ti.static(range(DIMENSIONS)):
            target[i][j] = source[i, j]

def render_3D(frame):
    if frame == 0 or frame == 1:
         create_flume_indices()
         create_flume_vertices()

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