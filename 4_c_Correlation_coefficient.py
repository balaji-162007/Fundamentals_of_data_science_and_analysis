import matplotlib.pyplot as plt
import numpy as np
x = np.random.rand(100)
y = 2 * x + np.random.normal(0, 0.1, 100)
plt.scatter(x, y, alpha=0.7)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot: X vs Y')
plt.show()
