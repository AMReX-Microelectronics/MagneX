import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the power law function
def power_law(x, a, b, c):
    return a * np.power(x, b) + c

# Generate some sample data
x_data = [10**3, 20**3, 30**3];#, 40**3, 50**3];
y_data = [8.15, 8.38, 8.43]; #, 8.45, 8.453]; 

# Fit the power law function to the data
popt, pcov = curve_fit(power_law, x_data, y_data, maxfev=10000, p0=[1, 1, 1])

# Extract the optimized parameters
a_opt, b_opt, c_opt = popt

# Generate fitted curve using optimized parameters
x_fit = np.linspace(1, 80**3, 1000)
y_fit = power_law(x_fit, a_opt, b_opt, c_opt)

# Plot the original data and the fitted curve
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, 'r-', label='Fitted Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Power Law Fit')
plt.show()

# Display the optimized parameters
print("Optimized parameters:")
print("a =", a_opt)
print("b =", b_opt)
print("c =", c_opt)

