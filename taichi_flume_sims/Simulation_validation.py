import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy import fft

g = 9.80665  # Gravitational acceleration taken to be positive downward
data_path = "/Users/treygower/code-REU/Physics-Informed-ML/Flume/dataset/train.npz"
H = 1.2 # Amplitude Expected
h = 1.85 # Water Depth Tsunami


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
    
    base_y = np.max(particles[0, :, 1]) * 102.4 # Gets Baseline max y value for water height
    y_values = particles[:, :, 1] * 102.4 
    x_values = particles[:, :, 0] * 102.4 
    t_steps = particles.shape[0]
    #x_sorted = sort_x(x_vals)
    y_values = np.stack(y_values)
    x_values = np.stack(x_values)

    y_max = []
    x_max = []
    time = []
    dt = 0.016666666666666666 # hardcode dt in from metadata
    fps = 60
    # counter = 0 
    for t in range(0, t_steps, 30): # not necessary to get each time step
        tc = t 
        for i in range(len(y_values[tc])):
            if np.max(y_values[tc,i]) > base_y:
                y_max.append(y_values[tc,i]-base_y)
                x_max.append(x_values[tc,i])
   
        time.append(t/fps)
    time = np.asarray(time)
    y_max = np.asarray(y_max)
    x_max = np.asarray(x_max)
    #interp_func = interp1d(x_max, y_max)
    #y_smooth = interp_func(y_max)
    #print(y_smooth)
    sorted_indices = np.argsort(x_max)

    print(sorted_indices)
    #time_sorted = time[sorted_indices]
    wave_numerical = (time,x_max,y_max)

    return wave_numerical


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

def plot_free_surface(x, t, y, wave_numerical):
    """
    Plot the surface elevations from the piston soliton wave in the flume's body of water.

    Args:
        x: Spatial points (m)
        t: Time points (s)
        y: Analytical solution values at each spatial and time point (m)
    """
    plt.figure(figsize=(10, 6))
    num_lines = len(t)
  
    cmap = plt.get_cmap('viridis')
    cmap_numeric = plt.get_cmap('viridis').reversed()

    for i, ti in enumerate(t):
        color = cmap(i / (num_lines - 1))
        color_numeric = cmap_numeric(i / (num_lines - 1))
        plt.plot(x, y[:,i], c=color, label=f't = {ti:.2f} s', linewidth=2)
    
    plt.plot(x, wave_numerical[2], c=color_numeric, label=f't = {ti:.2f} s', linewidth=2)
    plt.xlim([0,np.max(x)])
    plt.xlabel('Position (m)', fontsize=12)
    plt.ylabel('Wave Elevation (Amplitude)', fontsize=12)
    plt.title(f'Analytical vs numerical Surface Elevation for a Soliton Wave', fontsize=14)
    #plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def free_surface_error(x, y_analytical, y_numerical):
    """
    Calculate and plot the absolute error between numerical and analytical solutions.

    Args:
    t (np.ndarray): Time points.
    y_analytical (np.ndarray): Analytical solution values at each time point.
    y_numerical (np.ndarray): Numerical solution values at each time point.
    """
    numerical_reshaped = y_numerical[:, np.newaxis]  # Shape becomes (5968, 1)
    error = numerical_reshaped - y_analytical  # Result will be (5968, 10)
    # Analyze the difference for each time step

    plt.figure(figsize=(12, 6))
    plt.plot(x, error, label='Absolute Error', color='g', linewidth=2)
    
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



wave_numerical = load_data()
t = wave_numerical[0] #np.linspace(0, 10, 30)
x = wave_numerical[1]#np.linspace(0, 90, len(wave_numerical[2]))

#print('Computing Analytical...')
y_analytical = np.array([analytical_soliton(x, ti) for ti in t]).T

#print('Plotting...')
plot_free_surface(x, t, y_analytical, wave_numerical) # Graph Free Surface values

#free_surface_error(x, y_analytical, wave_numerical[2])
