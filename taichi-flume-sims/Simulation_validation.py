import numpy as np
import matplotlib.pyplot as plt


g = -9.80665  # Gravitational acceleration

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
                H: Wave height (Amplitude)
                h: Water depth
                c: Wave advective speed
        """
        c = np.sqrt( g * ( self.h + self.H ) )  # Wave speed for shallow water
        Ks = ( 1 / self.h ) * np.sqrt( ( 3 * self.H ) / ( 4 * self.h ) )
        return self.H * np.cosh( Ks * ( x - c * t ) ) ** -2 # Using np.cosh^-2 since cosh = 1/sech

    def free_surface(self, t, y_analytical, y_numerical):
        # Plotting Surface Elevations from the Piston Soliton Wave in the Flumes body of water
        plt.figure(figsize=(10, 6))
        plt.plot(t, y_numerical, label='Numerical Solution', linestyle='--', color = 'b')
        plt.plot(t, y_analytical, label='Analytical Solution', linewidth=2, color = 'r')
        plt.xlabel('Time (t)')
        plt.ylabel('Wave Elevation (η)')
        plt.title('Numerical vs Analytical Surface Elevation for a Soliton Wave')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def free_surface_error(self, t, y_analytical, y_numerical):
        # Calculate and plot error
        error = np.abs(y_numerical - y_analytical)
        plt.figure(figsize=(10, 4))
        plt.plot(t, error, label='Absolute Error', color='g')
        plt.xlabel('Time (t)')
        plt.ylabel('Absolute Error')
        plt.title('Absolute Error between Numerical and Analytical Solutions')
        plt.legend()
        plt.grid(True)
        plt.show()

    def validate_simulation(self, wave_numerical_soln):
        """
        Main Function for producing numeric vs analytic graphs
        """
        y_analytical = np.zeros((len(wave_numerical_soln), 1)) # Allocate Memory for Analytical Soln

        # Need the initial max water particle position
        t = wave_numerical_soln[:, 0]
        x = wave_numerical_soln[:, 1]
        y_numerical = wave_numerical_soln[:, 2]
        y_analytical[:] = self.analytical_soliton(x, t) # Compute analytical along the flume in a column vector

        self.free_surface(t, y_analytical, y_numerical) # Graph Free Surface values
        self.free_surface_error(t, y_analytical, y_numerical)

        
