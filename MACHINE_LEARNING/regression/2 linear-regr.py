import numpy as np
import  matplotlib.pyplot as plt

# x_train = np.array([1,2])     # x_train is the input variable
# y_train = np.array([300,500]) # y_train is output variable
# m = x_train.shape[0]          # m is the number of training examples
# i = 0                         # i is the index for the  training example
#
# x_i = x_train[i]
# y_i = y_train[i]
#
# plt.scatter(x_train,y_train,marker='x',color='red')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Training Data')
#
# w,b = 200, 100
#
# def compute_model_output(x, w, b):
#     m = x.shape[0]
#     f_wb = np.zeros(m)
#     for i in range(m):
#         f_wb[i] = w * x[i] + b
#
#     return f_wb
#
# tmp_f_wb = compute_model_output(x_train, w, b,)
# plt.plot(x_train, tmp_f_wb, color='blue',label='Prediction')
#
# plt.show()


# with scikit learn
from sklearn.linear_model import LinearRegression # importing the linear regression model from sklearn
from sklearn.metrics import mean_squared_error    # importing the mean squared error function from sklearn

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) # Hours studied (input feature X)
y = np.array([18, 25, 36, 48, 39])           # Marks obtained (target y)

model = LinearRegression()                   # Create an object of the LinearRegression class
model.fit(x, y)                              # Train the model using the training data

print(model.coef_)                           # slope (weight, w)
print(model.intercept_)                      # intercept (bias, b)

print(f"\nEquation of the line:")
print(f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

new_x = np.array([10, 2, 4, 5, 8]).reshape(-1, 1) # New hours of study (unseen data)
predictions = model.predict(new_x)                # Predict marks for the new hours

print("\nPredictions:")
for hour, pred in zip(new_x, predictions):
    print(f"For {hour[0]} hours of study -> Predicted Marks = {pred:.2f}")

print("\nTraining Data Comparison:")
training_predictions = model.predict(x)          # Predict marks for the original training data

cost = mean_squared_error(y,training_predictions)
print(f"\nCost: {cost:.2f}\n")

for hour, actual, pred in zip(x, y, training_predictions):
    print(
        f"Hours = {hour[0]} | "
        f"Actual Marks = {actual} | "
        f"Predicted Marks = {pred:.2f}"
    )

plt.scatter(x, y, color="red", label="Training Data") # Plot the original training data
x_line = np.arange(1, 9).reshape(-1, 1)               # Create more x values so the regression line extends further

plt.plot(x_line, model.predict(x_line),         # Plot the regression line
         color="blue",
         label="Regression Line")

plt.scatter(new_x, predictions,                        # Plot the predicted points
            color="green",
            marker="x",
            s=100,
            label="Predicted Points")

plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.title("Linear Regression using Scikit-learn")
plt.legend()
# plt.show()