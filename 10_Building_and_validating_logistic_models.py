import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(0)

# Generate dataset
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Plot results
plt.figure(figsize=(10, 6))

# True labels
plt.scatter(
    X_test[:, 0], X_test[:, 1],
    c=y_test, cmap='coolwarm',
    edgecolors='k', s=100,
    label='True Labels'
)

# Predicted labels
plt.scatter(
    X_test[:, 0], X_test[:, 1],
    c=y_pred, marker='x',
    cmap='coolwarm', s=100,
    label='Predicted Labels'
)

plt.title("Logistic Regression Predictions")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

