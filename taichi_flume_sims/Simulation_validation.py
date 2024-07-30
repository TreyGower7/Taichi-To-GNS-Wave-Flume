import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

g = 9.80665  # Gravitational acceleration taken to be positive downward
data_path = "/Users/treygower/code-REU/Physics-Informed-ML/Flume/dataset/train.npz"
H = 1.3 # Amplitude Expected
h = 2 # Water Depth Tsunami

def load_data():
    """Load data stored in npz format.

    The file format for Python 3.9 or less supports ragged arrays and Python 3.10
    requires a structured array. This function supports both formats.

    Args:
        path (str): Path to npz file.

    Returns:
        data (list): List of tuples of the form (positions, particle_type).
    """
    with np.load(data_path, allow_pickle=True) as data_file:
        if 'gns_data' in data_file:
            data = data_file['gns_data']
        else:
            data = [item for _, item in data_file.items()]
    
    particles = data[0][0]
    material_ids = data[0][1]
    
    base_y = np.max(particles[0, :, 1]) * 100 # Gets Baseline y value      
    
    # Filter water particles
    #wave_threshold =  # Particles where the wave height is greater than 1m above baseline (.029-.0194)

    y_values = particles[:, :, 1] * 100 # y-values are in second dimension
    x_values = particles[:, :, 0] * 100
    # Wave Particle Conditions
    water_cond = (material_ids == 5)[0] # Ensure Water Particle
    threshold_cond = y_values >= base_y # Ensure Wave is fully formed numerical soln
    wave_elevation_mask = water_cond & threshold_cond 
    y_filtered = y_values[wave_elevation_mask]
    x_filtered = x_values[wave_elevation_mask]
    #print(np.min(y_values))
    #print(np.min(y_filtered))
   
    wave_values = abs(y_filtered - base_y)
    # TODO:
    # Filter the maximal value for each wave to get the top most particles at each x positions
    # Save the x position of each top most particle for each wave along the surface of the water
    # Filter Duplicates to ensure plotting wave at each timestep

    #print(np.max(wave_values))
    print(wave_values.size)
    
    # Find peaks in the data
    peaks, _ = find_peaks(wave_values, distance=10)  # Adjust distance as needed

    downsampled_wave = wave_values[peaks]

    print(downsampled_wave.size)
    

    return downsampled_wave[::100]


def analytical_soliton(x, t):
    """ Analytical solution to non-periodic soliton wave for constant depth from:
        https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2008JC004932

        Args:
            x: Horizontal position in the flume
            t: Time step

        Other Vars:
            H: Wave height (Amplitude)
            h: Water depth
            c: Wave advective speed
    """
    c = np.sqrt( g * ( H + h ) )   # Wave speed for shallow water
    #c = np.sqrt( g * ( 1 + self.H / self.h ) )   # Wave speed for shallow water

    Ks = ( 1 / h ) * np.sqrt( ( 3 * H ) / ( 4 * h ) )
    return (H * np.cosh( Ks * ( x - c * t ) ) ** -2) # Using np.cosh^-2 since cosh = 1/sech

def plot_free_surface(x, t, y, y_numeric):
    """
    Plot the surface elevations from the piston soliton wave in the flume's body of water.

    Args:
        x: Spatial points (m)
        t: Time points (s)
        y: Analytical solution values at each spatial and time point (m)
    """
    plt.figure(figsize=(12, 8))
    num_lines = len(t)
  
    cmap = plt.get_cmap('viridis')
    cmap_numeric = plt.get_cmap('viridis').reversed()

    for i, ti in enumerate(t):
        color = cmap(i / (num_lines - 1))
        color_numeric = cmap_numeric(i / (num_lines - 1))
        plt.plot(x, y, c=color, label=f't = {ti:.2f} s', linewidth=2)
        plt.plot(x, y_numeric, c=color_numeric, label=f't = {ti:.2f} s', linewidth=2)

    plt.xlim([30,60])
    plt.xlabel('Position (m)', fontsize=12)
    plt.ylabel('Wave Elevation (Amplitude)', fontsize=12)
    plt.title(f'Analytical vs numerical Surface Elevation for a Soliton Wave', fontsize=14)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def free_surface_error(t, y_analytical, y_numerical):
    """
    Calculate and plot the absolute error between numerical and analytical solutions.

    Args:
    t (np.ndarray): Time points.
    y_analytical (np.ndarray): Analytical solution values at each time point.
    y_numerical (np.ndarray): Numerical solution values at each time point.
    """
    error = np.abs(y_numerical - y_analytical)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, error, label='Absolute Error', color='g', linewidth=2)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Absolute Error (m)', fontsize=12)
    plt.title('Absolute Error between Numerical and Analytical Solutions', fontsize=14)
    
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.yscale('log')  # Use log scale for better visualization
    
    plt.tight_layout()
    plt.show()

    # Some statistics about the error
    print(f"Maximum error: {np.max(error):.6f}")
    print(f"Mean error: {np.mean(error):.6f}")
    print(f"Standard deviation of error: {np.std(error):.6f}")



y_numerical = load_data()
t = np.linspace(0, 10, 10)
x = np.linspace(20, 90, y_numerical.size)

print('Computing Analytical...')
y_analytical = np.array([analytical_soliton(x, ti) for ti in t]).T

print('Plotting...')
plot_free_surface(x, t, y_analytical, y_numerical) # Graph Free Surface values
#plot_free_surface(x_values, t, y_numerical, "Numerical") # Graph Free Surface values

#free_surface_error(t, y_analytical, y_numerical)

