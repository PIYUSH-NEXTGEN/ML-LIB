# K-Nearest Neighbours (KNN)

K-Nearest Neighbours is one of the simplest machine learning algorithms that exists. The idea behind it is almost embarrassingly intuitive: to predict something about a new data point, look at the K most similar points in your training data and let them vote on the answer.

That is genuinely the entire algorithm. There is no training phase in the traditional sense, no weight updates, no gradient descent. The model just memorises the entire training dataset and does all its work at prediction time.

This makes KNN what is called a **lazy learner** — it does nothing during training and everything during inference.

---

## The Core Idea

Imagine you move to a new city and want to guess what neighbourhood you are in based on what the area looks like. You look around at your K nearest neighbours — the houses closest to you — and whichever neighbourhood most of them belong to, that is your best guess for your own neighbourhood.

KNN works exactly the same way in feature space. Every training example is a point in a multi-dimensional space. When a new point arrives, the algorithm finds the K training points closest to it and makes a prediction based on what those neighbours say.

```
Training data plotted in 2D feature space:

    x2
     |
     |   o o                    New point arrives: *
     |  o   o    * ?
     |   o o        x x
     |          x x   x
     |            x x
     |_________________________________ x1

     o = class A      x = class B

With K=3, find the 3 nearest neighbours to *
If 2 are class A and 1 is class B  →  predict class A
```

---

## How KNN Works Step by Step

**Step 1:** Choose the value of K (how many neighbours to consider).

**Step 2:** A new data point arrives.

**Step 3:** Calculate the distance between this new point and every single point in the training set.

**Step 4:** Sort all training points by distance and pick the K closest ones.

**Step 5:** For classification — take a majority vote among the K neighbours and assign that class. For regression — take the average of the K neighbours values and output that.

**Step 6:** That is the prediction. Done.

```
New point: [size=1400, rooms=3]

Training set distances:
  House A [1350, 3]  →  distance = 50.2   (nearest)
  House B [1500, 4]  →  distance = 101.5
  House C [1200, 2]  →  distance = 206.1  (third nearest)
  House D [900,  2]  →  distance = 502.0
  ...

With K=3, neighbours are: A, B, C

Classification: A=cheap, B=expensive, C=cheap  →  predict "cheap" (majority)
Regression:     A=₹45L, B=₹72L, C=₹38L        →  predict ₹51.7L (average)
```

---

## Distance Metrics

The heart of KNN is the distance calculation. How you measure "closeness" completely determines which neighbours get selected, which determines the prediction. Choosing the right distance metric matters.

### Euclidean Distance

The straight-line distance between two points. This is the default and most commonly used.

$$
d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}
$$

For two points in 2D with features $[x_1, x_2]$ and $[y_1, y_2]$:

$$
d = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}
$$

Works well when features are continuous and have similar scales. Sensitive to scale — a feature with values in the thousands will dominate the distance calculation over a feature with values between 0 and 1.

### Manhattan Distance

The sum of absolute differences along each dimension. Named after the grid-like street layout of Manhattan — you can only travel along axes, not diagonally.

$$
d(p, q) = \sum_{i=1}^{n} |p_i - q_i|
$$

More robust to outliers than Euclidean because large differences are not squared. Works better in high-dimensional spaces.

### Minkowski Distance

A generalisation that covers both Euclidean and Manhattan as special cases.

$$
d(p, q) = \left(\sum_{i=1}^{n} |p_i - q_i|^r\right)^{1/r}
$$

When $r = 1$ this is Manhattan distance. When $r = 2$ this is Euclidean distance.

### Hamming Distance

Used for categorical features. Counts how many positions are different between two data points.

```
Point A: [male, smoker, diabetic]
Point B: [male, non-smoker, diabetic]

Hamming distance = 1  (only the second feature differs)
```

### Which Distance to Use

| Situation | Recommended Distance |
|-----------|---------------------|
| Continuous numerical features | Euclidean (default) |
| High-dimensional data or outliers present | Manhattan |
| Mix of different feature types | Minkowski with tuned r |
| Categorical features | Hamming |

---

## Choosing K — The Most Important Decision

K is the only real hyperparameter in KNN, but it has a massive effect on the model's behaviour.

### Small K

With a very small K (say K=1), the model uses only the single nearest neighbour. This creates a very jagged, irregular decision boundary that closely follows every individual training point.

The model is highly sensitive to noise. A single mislabelled or unusual training point can throw off predictions for nearby test points. This is classic **overfitting** — too much attention paid to individual examples.

### Large K

With a very large K (say K=100 out of 200 training points), the model considers half the training set for every prediction. The decision boundary becomes very smooth.

But now the model may be too general. It ignores local patterns and predicts the majority class for almost everything. This is **underfitting** — not enough attention to local structure.

### The Sweet Spot

```
K too small (K=1):             K too large (K=100):
  jagged boundary                smooth but too simple
  overfitting                    underfitting
  low train error                high train error
  high test error                high test error

K just right:
  smooth but locally sensitive
  low train error
  low test error
```

### How to Choose K in Practice

**Rule of thumb:** Start with $K = \sqrt{n}$ where $n$ is the number of training examples. If you have 400 training points, try K=20 first.

**Cross-validation:** Try multiple values of K (e.g. 1, 3, 5, 7, 11, 15, 21) and evaluate each on a validation set. Pick the K that gives the lowest validation error.

**Odd vs even K:** For binary classification, always use an odd K to avoid ties.

---

## KNN for Classification vs Regression

KNN works for both types of supervised learning tasks with a small change in the final step.

### KNN Classification

After finding the K nearest neighbours, count how many belong to each class. Assign the new point to the class with the most votes.

$$
\hat{y} = \text{mode}(y_{k_1}, y_{k_2}, \ldots, y_{k_K})
$$

**Example:** K=5, neighbours are: [cat, cat, dog, cat, bird] → predict cat (3 out of 5 votes)

**Weighted voting:** You can give closer neighbours more weight in the vote. A neighbour at distance 1 should influence the prediction more than one at distance 100.

$$
\text{weight} = \frac{1}{d^2}
$$

### KNN Regression

After finding the K nearest neighbours, take the average (or weighted average) of their output values.

$$
\hat{y} = \frac{1}{K} \sum_{i=1}^{K} y_{k_i}
$$

**Example:** K=3, neighbours have prices [₹45L, ₹52L, ₹49L] → predict ₹48.67L

---

## Why Feature Scaling Is Absolutely Essential for KNN

KNN computes distances directly from raw feature values. If features have very different scales, the ones with larger values will completely dominate the distance calculation, and the other features will be ignored.

**Example without scaling:**

```
Feature 1: house size in sq ft  →  ranges from 500 to 5000
Feature 2: number of rooms      →  ranges from 1 to 10

Distance between house A [1500, 3] and house B [1600, 8]:
Euclidean = sqrt((1500-1600)² + (3-8)²)
          = sqrt(10000 + 25)
          = sqrt(10025)
          ≈ 100.1

The 5-room difference contributes only 0.25% of the total distance.
Number of rooms is effectively invisible to the model.
```

After standardisation, both features contribute equally. This is not optional in KNN — it is required.

---

## The Curse of Dimensionality

KNN works beautifully in low dimensions but starts breaking down as the number of features grows. This is called the curse of dimensionality.

In high-dimensional spaces, all points become approximately the same distance from each other. When every point is roughly equidistant, the concept of "nearest neighbours" loses its meaning — there is no clear distinction between near and far.

Concretely: in 1D space, the nearest neighbour of a point is genuinely close. In 1000D space, the nearest neighbour might still be very far away in absolute terms, and the difference in distance between the first and tenth nearest neighbour is tiny.

**Practical consequence:** KNN tends to perform poorly when you have more than 20 to 30 features without aggressive dimensionality reduction beforehand.

---

## KNN Has No Training Phase

This is worth emphasising because it is unusual. Most models do work during training (fitting parameters) and then make predictions quickly. KNN is the opposite.

| Phase | Most Algorithms | KNN |
|-------|----------------|-----|
| Training | Learns parameters (slow) | Just stores the data (instant) |
| Prediction | Apply learned parameters (fast) | Compute distances to all training points (slow) |
| Memory | Stores only parameters | Stores entire training set |

This has real practical implications. If your training set has 10 million points and 50 features, every single prediction requires computing 10 million distances. This is expensive. For large datasets, KNN becomes impractically slow at inference time without approximate nearest neighbour techniques like KD-trees or ball trees.

---

## When to Use KNN

KNN works well when:

- The dataset is small to medium sized (a few thousand to tens of thousands of examples)
- You have relatively few features (low to moderate dimensionality)
- The decision boundary is naturally irregular and non-linear
- You need a quick baseline model without much tuning
- Interpretability matters — you can always show which neighbours drove a prediction

KNN is not a good choice when:

- The dataset is very large (inference becomes too slow)
- You have many features (curse of dimensionality)
- Features are on very different scales and you cannot or forget to scale
- Real-time low-latency prediction is required
- You have lots of irrelevant or noisy features

---

## KNN in Code

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Always scale before KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Classification
model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

# Regression
model = KNeighborsRegressor(n_neighbors=5, metric='euclidean')
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)
```

Finding the best K using cross-validation:

```python
from sklearn.model_selection import cross_val_score
import numpy as np

k_values = [1, 3, 5, 7, 11, 15, 21]
cv_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"Best K: {best_k}")
```

---

## Quick Concept Summary

| Concept | What It Means |
|---------|--------------|
| K | Number of nearest neighbours to consider for each prediction |
| Lazy learner | No training phase, all computation happens at prediction time |
| Euclidean distance | Straight-line distance, the default metric for KNN |
| Manhattan distance | Sum of absolute differences, more robust to outliers |
| Majority vote | How KNN classifies: the most common class among K neighbours wins |
| Averaging | How KNN regresses: the mean of K neighbours values |
| Feature scaling | Mandatory for KNN, prevents large-valued features dominating distance |
| Curse of dimensionality | In high dimensions, all points become equidistant and KNN breaks down |
| Overfitting (small K) | Model too sensitive to individual points, jagged decision boundary |
| Underfitting (large K) | Model too general, smooth boundary that misses local patterns |
| Cross-validation | Standard method to find the optimal value of K |
