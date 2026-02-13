import matplotlib.pyplot as plt

# Data
a = [1, 2, 3, 4, 5]
b = [0, 0.6, 0.2, 15, 10, 8, 16, 21]
c = [4, 2, 6, 8, 3, 20, 13, 15]

# Plot lines
plt.plot(a, label='1st Rep')
plt.plot(b, "or", label='2nd Rep')   # o = circle, r = red
plt.plot(list(range(0, 22, 3)), label='3rd Rep')
plt.plot(c, label='4th Rep')

# Naming axes
plt.xlabel('Day ->')
plt.ylabel('Temp ->')

# Get current axes
ax = plt.gca()

# Hide top and right borders
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Set left boundary limits
ax.spines['left'].set_bounds(-3, 40)

# Set x-axis ticks
plt.xticks(list(range(-3, 10)))

# Set y-axis ticks
plt.yticks(list(range(-3, 20, 3)))

# Legend
plt.legend()

# Annotation
plt.annotate('Temperature V / s Days', xy=(1.01, -2.15))

# Title
plt.title('All Features Discussed')

# Show plot
plt.show()
