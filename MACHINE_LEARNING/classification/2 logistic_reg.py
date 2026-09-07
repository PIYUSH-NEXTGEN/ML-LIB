import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

# Hours studied
x = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)

# 0 = Fail
# 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(x, y)

print(f"\nCoefficient: {np.round(model.coef_, 2)}")
print(f"Intercept: {np.round(model.intercept_, 2)}")

prediction = model.predict([[5]])
print(f"Prediction for 5 hours of study: {prediction}")

probability = model.predict_proba([[5]])
print(f"Probability{np.round(probability, 2)}")

new_x = np.array([1,2,3,4,5,6,7]).reshape(-1,1)
predictions = model.predict(new_x)
print(f"Predictions for new data: {predictions}")

plt.scatter(x, y,
            color='red',
            label='Training Data')
x_line = np.linspace(1,7,100).reshape(-1,1)
probabilities = model.predict_proba(x_line)[:,1]

plt.plot(x_line,
         probabilities,
         color='blue',
         label='Sigmoid Curve')

plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.legend()
plt.show()







