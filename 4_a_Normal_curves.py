import numpy as np
import matplotlib.pyplot as plt
mu = 0
sigma = 1
data = np.random.normal(mu, sigma, 1000)
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma)**2)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Data values')
plt.ylabel('Probability density')
plt.title('Normal Distribution Curve')
plt.show()
