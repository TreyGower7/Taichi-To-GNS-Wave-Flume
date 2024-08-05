import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy import fft

g = 9.80665  # Gravitational acceleration taken to be positive downward
data_path = "/Users/treygower/code-REU/Physics-Informed-ML/Flume/dataset/train.npz"
data_path = "../Flume/dataset/train.npz"
data_path = "/home/justinbonus/SimCenter/Taichi-To-GNS-Wave-Flume/Flume/dataset/train.npz"
H = 1.0 # Amplitude Expected
h = 1.85 # Water Depth Tsunami


def load_data():
    """Load data stored in npz format

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
    base_y = 1.85
    y_values = particles[:, :, 1] * 102.4 
    x_values = particles[:, :, 0] * 102.4 
    t_steps = particles.shape[0]
    #x_sorted = sort_x(x_vals)
    y_values = np.stack(y_values)
    x_values = np.stack(x_values)

    x_max_all = []
    y_max_all = []
    
    time = []
    dt = 0.016666666666666666 # hardcode dt in from metadata
    fps = 60
    # counter = 0 
    print("T_Steps: ", t_steps)
    for t in range(0, t_steps, 30): # not necessary to get each time step
        tc = t 
        x_max = []
        y_max = []
        for i in range(len(y_values[tc])):
            if np.max(y_values[tc,i]) >= base_y:
                dx = 0.05
                offset = 3 * dx
                y_max.append(y_values[tc,i]-base_y - offset)
                x_max.append(x_values[tc,i])
   
        x_max = np.asarray(x_max)
        y_max = np.asarray(y_max)
        sorted_indices = np.argsort(x_max)

        x_max_all.append(x_max[sorted_indices])
        y_max_all.append(y_max[sorted_indices])
        time.append(t/fps)
        
    time = np.asarray(time)

    #interp_func = interp1d(x_max, y_max)
    #y_smooth = interp_func(y_max)
    #print(y_smooth)

    print(sorted_indices)

    #x_max_sorted = x_max[sorted_indices]
    #y_max_sorted = y_max[sorted_indices]
    print("Time Sorted: ", time)
    print("X Sorted: ", x_max_all[0])  
    print("Y Sorted: ", y_max_all[0])
    #print("Time Sorted: ", time_sorted)
    #print("X Sorted: ", x_max_sorted)
    #print("Y Sorted: ", y_max_sorted)
    wave_numerical = (time,x_max_all,y_max_all)

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


def hl_envelopes_idx(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax

def plot_free_surface(t, x, y, wave_numerical):
    """
    Plot the surface elevations from the piston soliton wave in the flume's body of water.

    Args:
        x: Spatial points (m)
        t: Time points (s)
        y: Analytical solution values at each spatial and time point (m)
    """
    
    from scipy.signal import savgol_filter
    
    # print("Min x", min(wave_numerical[1][0]))
    
    plt.figure(figsize=(10, 6))
    num_lines = len(t)
  
    cmap = plt.get_cmap('viridis')
    cmap_numeric = plt.get_cmap('viridis').reversed()

    for i, ti in enumerate(t):
        
        
        
        color = cmap(i / (num_lines - 1))
        color_numeric = cmap_numeric(i / (num_lines - 1))
        plt.plot(x, y[:,i], c=color, linewidth=1.5, linestyle='-')
        
        
        window = 5
        polyorder = 2
        if (wave_numerical[2][i].shape[0] >= window):
            dmax_window = 4
            lmin, lmax = hl_envelopes_idx(wave_numerical[2][i], dmin=1, dmax=dmax_window, split=False)
            y_numerical_filtered = wave_numerical[2][i][lmax]
            x_numerical_filtered = wave_numerical[1][i][lmax]
            # y_numerical_filtered = savgol_filter(y_numerical_filtered, 5, polyorder)
        else:
           y_numerical_filtered = wave_numerical[2][i]
           x_numerical_filtered = wave_numerical[1][i]

        dx = 0.05        
        taichi_offset = 3 * dx
        # if i % 2 == 0:
        plt.plot(x_numerical_filtered - taichi_offset, y_numerical_filtered,  c=color, label=f't = {ti:.2f} s', linewidth=1.5)
        # else:
            # plt.plot(x_numerical_filtered - taichi_offset, y_numerical_filtered,  c=color, linewidth=1.5)


    plt.xlim([0,np.max(x) / 2])
    plt.xlabel('Streamwise Position in Flume (m)', fontsize=12)
    plt.ylabel('Free-Surface Elevation Change of Wave (m)', fontsize=12)
    plt.title(f'Analytical vs Simulated Soliton Wave Free-Surface in the OSU LWF', fontsize=14)
    plt.legend(fontsize=10,  title='Time', bbox_to_anchor=(1.0125, 1.0), loc="upper left" )#, ncol =len(t))
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
t = wave_numerical[0] # np.linspace(0, 10, 30)
x = wave_numerical[1] # np.linspace(0, 90, len(wave_numerical[2]))

#print('Computing Analytical...')

# time_analytical = np.linspace(0, 10, 1000)
x_analytical = np.linspace(0, 90, 1000)
print("X Numerical Time-sets: ", len(x))
y_analytical = []
for i, ti in enumerate(t):
    print("Time: ", ti)
    print("i: ", i)
    piston_start_time = 1.0
    piston_motion_time = 2.5
    
    shifted_ti = ti -piston_start_time - piston_motion_time / 2 
    ya = analytical_soliton(x_analytical, shifted_ti)
    print("ya.shape: ", ya.shape)
    y_analytical.append(ya)

# print("X Analytical shape: ", x_analytical.shape)
# print("Y Numerical List: ", y_analytical)

y_analytical_list = np.asarray(y_analytical).T
print("Y Analytical List: ", y_analytical_list.shape)
print("Y Analytical List: ", y_analytical_list)
#print('Plotting...')
plot_free_surface(t, x_analytical, y_analytical_list, wave_numerical) # Graph Free Surface values

#free_surface_error(x, y_analytical, wave_numerical[2])
