import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate sample data (similar to your output image)
np.random.seed(42)
X = np.linspace(0, 10, 100)
Y = 2.5 * X + np.random.normal(0, 2, 100)

# ----------- First Plot (Scatter Only) -----------
plt.scatter(X, Y)
plt.title("Scatter Plot: X vs Y")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# ----------- Linear Regression -----------
X = X.reshape(-1, 1)

model = LinearRegression()
model.fit(X, Y)

slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope (beta_1): {slope}")
print(f"Intercept (beta_0): {intercept}")

# ----------- Fitted Line Plot -----------
Y_pred = model.predict(X)

plt.scatter(X, Y, label="Data")
plt.plot(X, Y_pred, color='red', label="Fitted Line")
plt.title("Simple Linear Regression: Fitted Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


# ----------- R-squared -----------
r_squared = model.score(X, Y)
print(f"R-squared: {r_squared}")

# ----------- Prediction -----------
X_new = np.array([[15]])
Y_new = model.predict(X_new)

print(f"Predicted Y for X = 15: {Y_new[0]}")
