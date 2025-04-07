import matplotlib.pyplot as plt
import numpy as np

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(12, 8))  # Adjust figure size as needed

# Plot the data
plt.plot(x, y, label='sin(x)', linewidth=2)

# Add labels and title with very large font sizes
plt.xlabel('X-axis', fontsize=48)
plt.ylabel('Y-axis', fontsize=48)
plt.title('Sine Wave', fontsize=60)

# Add a legend with a large font size
plt.legend(fontsize=36)

# Customize tick labels
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)


# Add grid
plt.grid(True)

# Save the figure (optional)
plt.savefig('large_font_plot.png')

#
