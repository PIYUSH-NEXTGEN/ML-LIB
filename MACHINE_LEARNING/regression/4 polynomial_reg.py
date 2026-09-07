import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x = np.array([1, 2, 3, 4, 5,7]).reshape(-1, 1)
y = np.array([2, 5, 10, 17, 26,30])

# Create an object that will convert x into polynomial features.
# degree = 2 means the model will use: 1, x, x²
poly = PolynomialFeatures(degree=2)

# Convert the original x into polynomial features.Columns represent: [Bias, x, x²]
x_poly = poly.fit_transform(x)
print(x_poly)

model = LinearRegression()  # Create an object of the LinearRegression class
model.fit(x_poly, y)        # Train model using the polynomial features it now learns: y = b + w₁x + w₂x²

print(f"\nCoefficient: {np.round(model.coef_, 2)}")
print(f"Intercept: {round(model.intercept_, 2)}")

new_x = np.array([[6]])
new_x_poly = poly.transform(new_x)    # Convert the new input into polynomial features
prediction = model.predict(new_x_poly)# Predict the output using the trained model
print(f"\nPrediction: {prediction}")

# Generate 100 equally spaced x-values between 1 and 5. These points make the curve smooth
x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)

x_line_poly = poly.transform(x_line) # Convert those x-values into polynomial features

plt.scatter(x, y,
            color='red',
            label='Training Data')

# Plot the polynomial regression curve
plt.plot(x_line,
         model.predict(x_line_poly),
         color='blue',
         label='Polynomial Regression Curve')

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

