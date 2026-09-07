# Gradient Descent 
Gradient Descent is an optimization algorithm used to find the best values of model parameters (such as `w` and `b`) by minimizing the cost (error) function.

---

## Real-Life Analogy

Imagine standing on a mountain in the dark and trying to reach the lowest point.

1. Feel the slope beneath your feet.
2. Take a small step downhill.
3. Repeat until you reach the valley.

The valley represents the minimum cost.

---

## Why Do We Need It?

For Linear Regression:

[
f(x) = wx + b
]

We need to find the best values of:

* `w` (slope)
* `b` (intercept)

that minimize prediction error.

---

## Cost Function

The cost function measures how bad the predictions are:

[
$J(w,b)=\frac{1}{2m}\sum_{i=1}^{m}(f(x_i)-y_i)^2$
]

* High Cost → Poor predictions
* Low Cost → Better predictions
* Cost = 0 → Perfect predictions

---

## Main Idea

Gradient Descent repeatedly:

1. Makes predictions
2. Calculates errors
3. Computes gradients (slopes)
4. Updates parameters
5. Repeats until cost is minimized

---

## Gradient

A gradient tells us:

> Which direction increases the cost the fastest?

Since we want to decrease cost, we move in the opposite direction of the gradient.

---

## Update Rule

For parameter `w`:

[
$w = w - \alpha \frac{\partial J}{\partial w}$
]

For parameter `b`:

[
$b = b - \alpha \frac{\partial J}{\partial b}$
]

Where:

* `α` = Learning Rate
* `∂J/∂w` = Gradient with respect to `w`
* `∂J/∂b` = Gradient with respect to `b`

---

## Learning Rate (α)

Controls step size during optimization.

### Too Small

* Very slow learning
* Takes many iterations

### Too Large

* Overshoots the minimum
* May never converge

### Good Learning Rate

* Reaches minimum efficiently
* Stable convergence

Common values:

```python
0.1
0.01
0.001
```

---

## Linear Regression Gradients

For:

[
f(x)=wx+b
]

Gradient for `w`:

[
$\frac{\partial J}{\partial w}$
=============================

$\frac{1}{m}
\sum_{i=1}^{m}
(f(x_i)-y_i)x_i$
]

Gradient for `b`:

[
$\frac{\partial J}{\partial b}$
=============================

$\frac{1}{m}
\sum_{i=1}^{m}
(f(x_i)-y_i)$
]
---
# Mathematical Example of Gradient Descent

## Problem

Suppose we want to minimize the function:

[
J(w) = w^2
]

The minimum value occurs at:

[
w = 0
]

Let's see how Gradient Descent finds it.

---

## Step 1: Find the Gradient

Differentiate the cost function:

[
$J(w)=w^2$
]

[
$\frac{dJ}{dw}=2w$
]

This tells us the slope at any value of (w).

---

## Step 2: Choose Initial Values

Let's start with:

```text
w = 4
Learning Rate (α) = 0.1
```

---

## Step 3: Apply Gradient Descent

Update Rule:

[
$w = w - \alpha \frac{dJ}{dw}$
]

Substitute values:

[
$w = 4 - 0.1(2 \times 4)$
]

[
w = 4 - 0.8
]

[
w = 3.2
]

---

## Iteration 2

Current:

[
w = 3.2
]

Gradient:

[
2w = 6.4
]

Update:

[
w = 3.2 - 0.1(6.4)
]

[
w = 2.56
]

---

## Iteration 3

Current:

[
w = 2.56
]

Gradient:

[
2w = 5.12
]

Update:

[
w = 2.56 - 0.1(5.12)
]

[
w = 2.048
]

---

## Progress Table

| Iteration | w      |
| --------- | ------ |
| 0         | 4.0000 |
| 1         | 3.2000 |
| 2         | 2.5600 |
| 3         | 2.0480 |
| 4         | 1.6384 |
| 5         | 1.3107 |
| 10        | 0.4295 |
| 20        | 0.0461 |

---

## Observation

Notice that:

* (w) gets smaller every iteration.
* The updates become smaller as we approach the minimum.
* Eventually:

[
$w \rightarrow 0$
]

which is exactly the minimum of:

[
$J(w)=w^2$
]

---

## Types of Gradient Descent

### 1. Batch Gradient Descent

Uses the entire dataset before updating parameters.

* Accurate
* Slow on large datasets

### 2. Stochastic Gradient Descent (SGD)

Updates after every training example.

* Fast
* Noisy updates

### 3. Mini-Batch Gradient Descent

Updates after small batches (32, 64, 128 samples, etc.).

* Fast
* Stable
* Most commonly used in Deep Learning

---


---

## Key Takeaway

Gradient Descent is the foundation of:

* Linear Regression
* Logistic Regression
* Neural Networks
* Deep Learning
* Large Language Models (LLMs)

---
# Gradient Descent for Multiple Linear Regression 

When you have **more than one input feature**, you use Multiple Linear Regression:

$$
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

**Example — Predicting House Price:**

| Feature | Symbol | Value |
|---------|--------|-------|
| Size (sq ft) | $x_1$ | 1500 |
| Number of rooms | $x_2$ | 3 |
| Location score | $x_3$ | 8 |
| Age (years) | $x_4$ | 10 |

$$
\hat{y} = w_1(1500) + w_2(3) + w_3(8) + w_4(10) + b
$$

Instead of learning one weight $w$, the model must now learn **$n$ weights**  one per feature plus the bias $b$.

Find the values of $w_1, w_2, \ldots, w_n$ and $b$ that **minimise the cost function**:

$$
J(w_1, w_2, \ldots, w_n, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$$

The only difference from simple linear regression is that the cost now depends on **multiple weights**, not just one.

---

## Gradient Descent Update Rules

In simple linear regression, you updated one weight $w$ and one bias $b$.

In multiple linear regression, you update **every weight separately**, plus the bias — all in the same step.

### Update Rule for Each Weight $w_j$

$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j} \quad \text{for } j = 1, 2, \ldots, n
$$

### Update Rule for Bias $b$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

Where $\alpha$ is the **learning rate**  how big each step is.

---

## The Gradients

After computing the partial derivatives of the MSE cost function, the gradients work out to:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) \cdot x_j^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)
$$

Where:
- $m$ = number of training examples
- $\hat{y}_i - y_i$ = prediction error for the $i$-th example
- $x_j^{(i)}$ = value of feature $j$ for the $i$-th example

### What This Is Saying

Each weight $w_j$ is updated based on:
- **How wrong the predictions are** $(\hat{y}_i - y_i)$
- **How much feature $j$ contributed** to those predictions $(x_j^{(i)})$

If a feature consistently contributes to large errors, its weight gets adjusted more aggressively.

---

## Full Update Step Written Out

With $n$ features, one complete gradient descent step looks like this:

$$
w_1 := w_1 - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) x_1^{(i)}
$$

$$
w_2 := w_2 - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) x_2^{(i)}
$$

$$
\vdots
$$

$$
w_n := w_n - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i) x_n^{(i)}
$$

$$
b := b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)
$$

All weights and bias are updated **simultaneously**  you compute all the gradients first using the current values, then apply all the updates at once.

---

## Simultaneous Update

All parameters must be updated **at the same time**, using the gradients computed from the **same current values**.

```
✅ Correct — simultaneous update:
   temp_w1 = w1 - α * gradient_w1
   temp_w2 = w2 - α * gradient_w2
   temp_b  = b  - α * gradient_b
   w1 = temp_w1
   w2 = temp_w2
   b  = temp_b

❌ Wrong — sequential update:
   w1 = w1 - α * gradient_w1   ← w1 already changed
   w2 = w2 - α * gradient_w2   ← gradient used old w1, but w1 is now different
   b  = b  - α * gradient_b
```

Updating sequentially causes each gradient to be computed from a partially-updated model  which is mathematically inconsistent and leads to incorrect results.

---

## Vectorized Implementation

Writing out a separate update for every weight is impractical when $n = 100$ or $n = 1000$. Vectorization handles all weights in one line.

### Vectorized Prediction

$$
\hat{Y} = Xw + b
$$

```python
y_hat = np.dot(X, w) + b        # shape: (m,)
```

### Vectorized Gradients

$$
\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{Y} - Y)
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum (\hat{Y} - Y)
$$

```python
errors    = y_hat - y                        # shape: (m,)
grad_w    = np.dot(X.T, errors) / m         # shape: (n,)
grad_b    = np.sum(errors) / m              # scalar
```

### Vectorized Parameter Update

```python
w = w - alpha * grad_w
b = b - alpha * grad_b
```

All $n$ weights updated in one line. No loop over features needed.

---

## Full Training Loop

```python
# Initialise
w = np.zeros(n)      # one weight per feature
b = 0
alpha = 0.01
iterations = 1000

for _ in range(iterations):

    # Step 1: Predict
    y_hat = np.dot(X, w) + b

    # Step 2: Compute gradients
    errors = y_hat - y
    grad_w = np.dot(X.T, errors) / m
    grad_b = np.sum(errors) / m

    # Step 3: Update parameters simultaneously
    w = w - alpha * grad_w
    b = b - alpha * grad_b
```

---

## Simple vs Multiple — Key Differences

| | Simple Linear Regression | Multiple Linear Regression |
|--|--------------------------|---------------------------|
| **Features** | 1 | $n$ (many) |
| **Weights to learn** | 1 ($w$) | $n$ ($w_1, w_2, \ldots, w_n$) |
| **Prediction** | $\hat{y} = wx + b$ | $\hat{y} = w_1x_1 + \cdots + w_nx_n + b$ |
| **Gradient for weights** | One gradient | One gradient per weight |
| **Update step** | Update $w$ and $b$ | Update all $w_j$ and $b$ simultaneously |
| **Cost function** | Same MSE | Same MSE |
| **Gradient Descent logic** | Identical | Identical — just more weights |

The math is exactly the same. Multiple linear regression is just simple linear regression **scaled up to $n$ features**.

---

## Quick Concept Summary

| Concept | Meaning |
|---------|---------|
| $w_j$ | Weight for the $j$-th feature |
| $\frac{\partial J}{\partial w_j}$ | How much the cost changes if $w_j$ changes slightly |
| Simultaneous update | Compute all gradients first, then apply all updates at once |
| Vectorization | Replace per-weight loops with matrix operations |
| $X^T(\hat{Y} - Y)$ | Vectorized gradient — computes all $n$ weight gradients at once |
| Learning rate $\alpha$ | Step size — same role as in simple linear regression |
