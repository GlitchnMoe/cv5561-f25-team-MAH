import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def mandelbrot(c, max_iter):
    """
    Calculate the number of iterations for a complex number c
    to escape the Mandelbrot set (or max_iter if it doesn't escape)
    """
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Compute the Mandelbrot set for a given region
    """
    # Create coordinate arrays
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    
    # Create output array
    mandelbrot_set = np.zeros((height, width))
    
    # Compute Mandelbrot set
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            mandelbrot_set[i, j] = mandelbrot(c, max_iter)
    
    return mandelbrot_set

# Set parameters
width, height = 800, 600
max_iter = 100

# Define the complex plane region to visualize
xmin, xmax = -2.5, 1.0
ymin, ymax = -1.25, 1.25

print("Computing Mandelbrot set...")
mandelbrot_set = compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)

# Create a custom colormap
colors = ['#000033', '#000055', '#0000BB', '#0E4C92', '#2E8BC0', 
          '#19D3F3', '#FFF700', '#FF8C00', '#FF0080', '#000000']
n_bins = 256
cmap = LinearSegmentedColormap.from_list('mandelbrot', colors, N=n_bins)

# Create the plot
plt.figure(figsize=(12, 9))
plt.imshow(mandelbrot_set, extent=[xmin, xmax, ymin, ymax], 
           cmap=cmap, interpolation='bilinear', origin='lower')
plt.colorbar(label='Iterations to escape', shrink=0.8)
plt.title('The Mandelbrot Set', fontsize=16, fontweight='bold')
plt.xlabel('Real axis')
plt.ylabel('Imaginary axis')
plt.tight_layout()

print("Displaying Mandelbrot set...")
plt.show()