import numpy as np
import matplotlib.pyplot as plt


g = 9.80665  # Gravitational acceleration taken to be positive downward

class SolitonWaveValidation:
    def __init__(self, H, h):
        self.H = H
        self.h = h

    def analytical_soliton(self, x, t):
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
        c = np.sqrt( g * ( self.H + self.h ) )   # Wave speed for shallow water
        #c = np.sqrt( g * ( 1 + self.H / self.h ) )   # Wave speed for shallow water

        Ks = ( 1 / self.h ) * np.sqrt( ( 3 * self.H ) / ( 4 * self.h ) )
        return self.H * np.cosh( Ks * ( x - c * t ) ) ** -2 # Using np.cosh^-2 since cosh = 1/sech

    def plot_free_surface(self, x, t, y, type_wave):
        """
        Plot the surface elevations from the piston soliton wave in the flume's body of water.

        Args:
            x: Spatial points (m)
            t: Time points (s)
            y: Analytical solution values at each spatial and time point (m)
        """
        plt.figure(figsize=(12, 8))
        num_lines = len(t)

        if type_wave == "Analytical":
            cmap = plt.get_cmap('viridis')
        else:
            cmap_num = plt.get_cmap('viridis').reversed

        for i, ti in enumerate(t):
            color = cmap(i / (num_lines - 1))
            plt.plot(x, y[:, i], color=color, label=f't = {ti:.2f} s', linewidth=2)

        plt.xlabel('Position (m)', fontsize=12)
        plt.ylabel('Wave Elevation (m)', fontsize=12)
        plt.title(f'{type_wave} Surface Elevation for a Soliton Wave', fontsize=14)
        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def free_surface_error(self, t, y_analytical, y_numerical):
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

    def validate_simulation(self, wave_numerical_soln):
        """
        Main Function for producing numeric vs analytic graphs
        """
        t = wave_numerical_soln[:, 0]
        x = wave_numerical_soln[:, 1]
        y_numerical = wave_numerical_soln[:, 2]

        # Generate time and space points
        ta = np.linspace(0, 10, 10)  # Time points (s)
        xa = np.linspace(0, 90, 1000)  # Spatial points (m)
        y_analytical = np.array([self.analytical_soliton(xa, ti) for ti in ta]).T

       
        self.plot_free_surface(xa, ta, y_analytical, "Analytical") # Graph Free Surface values
        #self.plot_free_surface(x, t, y_numerical, "Numerical") # Graph Free Surface values

        self.free_surface_error(t, y_analytical, y_numerical)

        