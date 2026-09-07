# Polynomial Regression 
Linear regression assumes the relationship between input and output is a straight line. But most real-world data doesn't behave that way.

Consider house prices. A small increase in size from 500 to 600 sq ft might add ₹2 lakhs. But the same 100 sq ft increase from 2000 to 2100 sq ft might add ₹8 lakhs. The relationship is not linear — it curves.

If you force a straight line through curved data, you get poor predictions no matter how long you train. Polynomial regression solves this by letting the model fit **curves** instead of just straight lines.

---

## What Is Polynomial Regression?

Polynomial regression extends linear regression by adding **higher-degree versions of the same feature** as new input columns.

Instead of:

$$
\hat{y} = w_1x + b
$$

You use:

$$
\hat{y} = w_1x + w_2x^2 + w_3x^3 + \cdots + w_nx^n + b
$$

Here $x^2$, $x^3$ and so on are not separate features from your dataset. They are just the **original feature raised to higher powers**, created artificially before training.

Once you create these columns, the rest of the training process is identical to multiple linear regression. The model learns weights for each term using the same cost function and gradient descent.

---

## A Concrete Example

Suppose you want to predict crop yield based on the amount of rainfall.

| Rainfall (x) | Yield (y) |
|-------------|----------|
| 10 | 20 |
| 20 | 45 |
| 30 | 80 |
| 40 | 95 |
| 50 | 85 |
| 60 | 60 |

Notice that yield goes up initially but then drops after too much rain (waterlogging). A straight line cannot capture this. A polynomial can.

With degree 2, you create a new feature $x^2$:

| $x$ (rainfall) | $x^2$ | $y$ (yield) |
|---------------|-------|------------|
| 10 | 100 | 20 |
| 20 | 400 | 45 |
| 30 | 900 | 80 |
| 40 | 1600 | 95 |
| 50 | 2500 | 85 |
| 60 | 3600 | 60 |

Now you train a multiple linear regression model on both columns. The model learns:

$$
\hat{y} = w_1x + w_2x^2 + b
$$

This gives you a curve that rises and then falls matching the actual data pattern.

---

## Degree of the Polynomial

The **degree** controls how complex the curve can be.

**Degree 1** — straight line (this is just regular linear regression)

$$
\hat{y} = w_1x + b
$$

**Degree 2** — one bend in the curve (parabola), good for data that rises and falls

$$
\hat{y} = w_1x + w_2x^2 + b
$$

**Degree 3** — two bends, can model more complex shapes

$$
\hat{y} = w_1x + w_2x^2 + w_3x^3 + b
$$

**Higher degrees** — more flexible, but increasingly dangerous (explained below)

As you increase the degree, the model gets more expressive. But more expressive is not always better.

---

## Feature Engineering Connection

Polynomial regression is actually a special case of **feature engineering** the process of creating new input columns from existing ones to help the model learn better.

You can also use other transformations depending on what the data looks like:

$$
\hat{y} = w_1x + w_2\sqrt{x} + b
$$

$$
\hat{y} = w_1x + w_2\log(x) + b
$$

These are not strictly polynomial, but the idea is the same. You look at the shape of your data, pick a transformation that matches it, create the new column, and train as usual.

The key insight is that the model itself stays linear in terms of how it learns — it's still just multiplying weights by features and summing. You're just choosing smarter features.

---

## The Overfitting Problem

This is the most important thing to understand about polynomial regression.

With a high enough degree, the model can fit the training data almost perfectly. But it starts memorising noise — tiny random fluctuations in the data — rather than learning the actual underlying pattern.

**Underfitting** (degree too low): The curve is too simple. It misses the real pattern. High error on both training and new data.

**Good fit** (degree just right): The curve follows the real pattern. Low error on training data and reasonable error on new data.

**Overfitting** (degree too high): The curve chases every data point including noise. Very low error on training data, but terrible on new data because it learned the noise, not the pattern.

---

## How to Choose the Right Degree

You don't just guess. You try different degrees and evaluate them properly.

**Step 1** — Split your data into training set and test set (or use cross-validation).

**Step 2** — Train models with degree 1, 2, 3, 4 and so on.

**Step 3** — Check the cost on both training and test data after each:

| Degree | Train Cost | Test Cost | Diagnosis |
|--------|-----------|----------|-----------|
| 1 | High | High | Underfitting |
| 2 | Low | Low | Good fit |
| 3 | Low | Low | Still good |
| 8 | Very Low | Very High | Overfitting |

Pick the degree where test cost is lowest. That is the degree that generalises best to data the model has never seen.

---

## Feature Scaling Is Essential Here

When you create polynomial features, the scale differences become extreme very quickly.

If $x$ ranges from 1 to 100:
- $x$ ranges from 1 to 100
- $x^2$ ranges from 1 to 10,000
- $x^3$ ranges from 1 to 1,000,000

Without scaling, gradient descent will struggle badly. The features are on completely different scales and the cost function becomes a badly shaped landscape.

Always apply standardisation or min-max scaling **after** creating the polynomial features and **before** training.

```
Original x  →  Create x², x³  →  Scale all columns  →  Train model
```

---

## Implementation in Code

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

x = np.array([10, 20, 30, 40, 50, 60]).reshape(-1, 1)
y = np.array([20, 45, 80, 95, 85, 60])

# Step 1: Create polynomial features (degree 2 gives x and x²)
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# Step 2: Scale the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_poly)

# Step 3: Train as usual
model = LinearRegression()
model.fit(x_scaled, y)
```

The `PolynomialFeatures` step does the feature creation. Everything after that is just standard linear regression.

---

## When to Use Polynomial Regression

**Use it when:**
- The data clearly curves when you plot it
- Linear regression gives high error even after tuning
- The relationship logically involves acceleration or diminishing returns (speed, growth curves, physical processes)

**Be careful when:**
- You have very little data — high degree polynomials overfit easily on small datasets
- You are extrapolating beyond the range of your training data — polynomial curves can go wildly wrong outside the data range
- You have many features — polynomial expansion creates a huge number of columns and can become computationally expensive very quickly

---

## Quick Concept Summary

| Concept | What It Means |
|---------|--------------|
| Polynomial features | New columns created by raising the original feature to higher powers |
| Degree | How many powers you add — controls curve flexibility |
| Underfitting | Degree too low, model too simple, misses the real pattern |
| Overfitting | Degree too high, model memorises noise, fails on new data |
| Feature scaling | Essential after creating polynomial features due to extreme scale differences |
| Still linear regression | The model training is identical — only the input features changed |
