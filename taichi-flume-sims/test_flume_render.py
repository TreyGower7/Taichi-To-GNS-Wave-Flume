import taichi as ti 
import numpy as np 

ti.init(arch=ti.gpu, default_fp=ti.f32)

# Define Taichi fields for flume geometry
flume_vertices = ti.Vector.field(3, dtype=float, shape=8)
bottom = ti.field(dtype=int, shape=6)  # Change to a 1D array
# Only way to render flume with differing colors
backwall = ti.field(dtype=int, shape=6)
frontwall = ti.field(dtype=int, shape=6)
sidewalls = ti.field(dtype=int, shape=6)

Flume_height_3d = 3.7 # meters
Flume_width_3d = 4.6 # meters
grid_length = 102.4
#168, 153, 50
front_back_color = (166/255, 126/255, 71/255)  # Rust color in RGB
side_color = (197/255, 198/255, 199/255)  # Light grey color in RGB
bottom_color = (79/255, 78/255, 78/255)
background_color = (237/255, 235/255, 235/255)

n_grid = 2048
gui_res = min(1080, n_grid) # Set the resolution of the GUI

@ti.kernel
def create_flume_vertices():
    flume_vertices[0] = ti.Vector([0, 0, 0])
    flume_vertices[1] = ti.Vector([grid_length, 0, 0])
    flume_vertices[2] = ti.Vector([grid_length, 0, Flume_width_3d])
    flume_vertices[3] = ti.Vector([0, 0, Flume_width_3d])
    flume_vertices[4] = ti.Vector([0, Flume_height_3d, 0])
    flume_vertices[5] = ti.Vector([grid_length, Flume_height_3d, 0])
    flume_vertices[6] = ti.Vector([grid_length, Flume_height_3d, Flume_width_3d])
    flume_vertices[7] = ti.Vector([0, Flume_height_3d, Flume_width_3d])

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

# Initialize flume geometry
create_flume_vertices()
create_flume_indices()

# Create window outside the render function
window = ti.ui.Window("Digital Twin of the NSF OSU LWF Facility - Tsunami Debris Simulation in MPM - 3D", res=(gui_res, gui_res))
canvas = window.get_canvas()
scene = window.get_scene()

camera = ti.ui.Camera()

def render_scene():
    
    # Camera positioned based on flume parameters
    camera.position(grid_length*1.2, Flume_height_3d*8, Flume_width_3d*6)
    
    # Camera looking at the center of the flume
    camera.lookat(grid_length/2, Flume_height_3d/2, Flume_width_3d/2)
    
    # Set the up direction as the y-axis for congruency with particles
    camera.up(0, 1, 0)

    camera.fov(60)

    # Set the camera for this frame
    scene.set_camera(camera)

    # Set up the light
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(grid_length/2, Flume_height_3d*2, Flume_width_3d*2), color=(1, 1, 1))

    # Render the flume
    # Render the bottom face
    scene.mesh(flume_vertices, bottom, color=bottom_color)
    
    # Render each side face separately (if only taichi supported slicing)
    scene.mesh(flume_vertices, backwall, color=front_back_color)
    scene.mesh(flume_vertices, sidewalls, color=side_color)
    scene.mesh(flume_vertices, frontwall, color=front_back_color)

    # Render the scene
    canvas.scene(scene)


# Main loop
while window.running:
    render_scene()
    window.show()