# Vectorization in Machine Learning

Vectorization is the practice of replacing **explicit loops** with **array operations** that process all values simultaneously.

Instead of computing one value at a time using a loop, you hand the entire array to a highly optimised mathematical library and let it do everything in one instruction.

```
Non-vectorized:  compute result[0], then result[1], then result[2] ...
Vectorized:      compute all results at once  →  done
```

### Simple Analogy

Imagine you need to multiply 1,000 numbers by 2.

- **Loop approach:** Pick up number 1, multiply by 2, put it down. Pick up number 2, multiply by 2, put it down. Repeat 1,000 times.
- **Vectorized approach:** Put all 1,000 numbers on a conveyor belt. One machine processes all of them simultaneously. Done in one pass.


---

## Why It Matters in Machine Learning

In ML, you're almost never working with a single data point. You have:

- Thousands to millions of training examples
- Dozens to hundreds of features per example
- Thousands of gradient descent iterations

At that scale, the difference between a loop and a vectorized operation is the difference between **minutes and hours**, or **hours and days**.

### Example Dot Product

A dot product multiplies two vectors element-wise and sums the results. This is the core operation behind nearly every ML algorithm.

Given:

w = [w_1, w_2, w_3], \quad x = [x_1, x_2, x_3]


$$
w \cdot x = w_1 x_1 + w_2 x_2 + w_3 x_3
$$

**Without vectorization (loop):**

```python
result = 0
for i in range(len(w)):
    result += w[i] * x[i]
```

**With vectorization (NumPy):**

```python
result = np.dot(w, x)
```

One line. No loop. NumPy handles it internally using optimised low-level code it runs orders of magnitude faster.

---

## Vectors, Matrices


### Scalar — A Single Number

```
x = 42
```

### Vector — A 1D Array of Numbers

A list of features for one training example, or a list of weights.

```
x = [1500, 3, 8, 10]   →  house: 1500 sqft, 3 rooms, location score 8, age 10 years
w = [300, 5000, 10000, -200]   →  learned weights for each feature
```

### Matrix — A 2D Array of Numbers

A collection of multiple training examples stacked together.

```
X = [
  [1500, 3, 8, 10],   ← house 1
  [900,  2, 6, 25],   ← house 2
  [2100, 4, 9, 5],    ← house 3
]
```

This matrix $X$ has shape $(3, 4)$ — 3 examples, 4 features each.

In ML, your entire dataset is typically one big matrix $X$, and the model operates on **all rows at once** using matrix operations.

---

## Vectorization in the ML Prediction Step

### Without Vectorization

For $m$ training examples with $n$ features, making predictions one at a time:

```python
predictions = []
for i in range(m):
    y_hat = 0
    for j in range(n):
        y_hat += w[j] * X[i][j]
    y_hat += b
    predictions.append(y_hat)
```

Two nested loops. Slow. 

### With Vectorization

```python
predictions = np.dot(X, w) + b
```

One line. NumPy multiplies the entire matrix $X$ by the weight vector $w$ and adds bias $b$ for all $m$ examples and $n$ features simultaneously.

Mathematically:

$$
\hat{Y} = Xw + b
$$

This is the **vectorized form** of the prediction equation for all examples at once.

---

## Vectorization in the Cost Function

### Non-Vectorized

```python
cost = 0
for i in range(m):
    error = predictions[i] - y[i]
    cost += error ** 2
cost = cost / (2 * m)
```

### Vectorized

```python
errors = predictions - y          # subtract entire arrays at once
cost = np.sum(errors ** 2) / (2 * m)
```

Or even more concisely:

```python
cost = np.mean((predictions - y) ** 2) / 2
```

NumPy operates on the entire array in each step — no loop needed.

---

## Vectorization in Gradient Descent

The gradient update for weights in linear regression:

$$
\frac{\partial J}{\partial w} = \frac{1}{m} X^T (Xw - y)
$$

### Non-Vectorized

```python
gradients = np.zeros(n)
for i in range(m):
    error = predictions[i] - y[i]
    for j in range(n):
        gradients[j] += error * X[i][j]
gradients = gradients / m
```

### Vectorized

```python
errors = predictions - y                   # shape: (m,)
gradients = np.dot(X.T, errors) / m       # shape: (n,)
```

The matrix transpose $X^T$ and a single dot product replaces the entire double loop. This is dramatically faster and easier to read.

---

## Why Vectorization Is So Fast

Three reasons vectorized operations are much faster than loops in Python:

### 1. Low-Level Optimised Code

NumPy operations are written in C under the hood. Python loops are interpreted one line at a time. C runs 10–100x faster than interpreted Python.

### 2. SIMD — Single Instruction, Multiple Data

Modern CPUs have special instructions that apply one operation to multiple values simultaneously. NumPy takes advantage of this — one CPU instruction processes 4, 8, or even 16 numbers at the same time.

```
Loop:       ADD x[0]   ADD x[1]   ADD x[2]   ADD x[3]   ← 4 separate instructions
SIMD:       ADD x[0], x[1], x[2], x[3]                  ← 1 instruction
```

### 3. GPU Parallelism (at larger scale)

GPUs contain thousands of small cores designed to perform the same operation on massive arrays simultaneously. Deep learning frameworks like PyTorch and TensorFlow use vectorized matrix operations that run on GPUs, enabling training on millions of examples in seconds rather than hours.

```
CPU: 8–16 cores  →  good for loops
GPU: 10,000+ cores  →  perfect for matrix operations
```

This is why vectorization is not just a convenience — it's what makes deep learning **possible at all**.

---

## Speed Comparison  Concrete Example

Multiplying two arrays of 1,000,000 numbers:

```python
import numpy as np
import time

a = np.random.rand(1_000_000)
b = np.random.rand(1_000_000)

# Loop approach
start = time.time()
result = 0
for i in range(len(a)):
    result += a[i] * b[i]
print("Loop time:", time.time() - start)     # ~0.5 seconds

# Vectorized approach
start = time.time()
result = np.dot(a, b)
print("Vectorized time:", time.time() - start)  # ~0.001 seconds
```

**Vectorized is ~500x faster.** At a million examples, that gap only grows.

---

## Broadcasting — Vectorization's Hidden Superpower

Broadcasting is a NumPy feature that lets you perform operations on arrays of **different shapes** without writing loops or manually copying data.

### Example

Add a bias value $b = 5$ to every element of a vector with 4 elements:

```python
x = np.array([10, 20, 30, 40])
b = 5

result = x + b   # →  [15, 25, 35, 45]
```

NumPy "broadcasts" the scalar $b$ across the entire array automatically. No loop needed.

### More Useful Example — Adding Bias to All Predictions

```python
X = np.array([          # shape: (3, 4)
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])
b = np.array([1, 0, -1, 2])   # shape: (4,)

result = X + b                 # shape: (3, 4) — b is added to every row
```

NumPy automatically stretches $b$ to match the shape of $X$. This removes entire categories of loops from ML code.

### Broadcasting Rules (Simple Version)

Two arrays can be broadcast together if, going from right to left, their dimensions either:
- Match exactly, or
- One of them is 1

```
Shape (3, 4) + Shape (4,)   →  valid → (3, 4)   ✅
Shape (3, 4) + Shape (3,)   →  invalid                 ❌
Shape (3, 4) + Shape (1, 4) →  valid → (3, 4)   ✅
Shape (3, 4) + Shape (3, 1) →  valid → (3, 4)   ✅
```

---

## Common NumPy Operations Used in ML

| Operation | Code | What It Does |
|-----------|------|--------------|
| Dot product | `np.dot(a, b)` | Element-wise multiply + sum |
| Matrix multiply | `np.matmul(A, B)` or `A @ B` | Full matrix multiplication |
| Element-wise multiply | `a * b` | Multiply arrays element by element |
| Transpose | `A.T` | Flip rows and columns |
| Sum | `np.sum(a)` | Sum all elements |
| Mean | `np.mean(a)` | Average of all elements |
| Squared | `a ** 2` | Square every element |
| Square root | `np.sqrt(a)` | Square root of every element |
| Exponential | `np.exp(a)` | $e^x$ for every element |

---

## Vectorized vs Loop — Side by Side Summary

| | Loop | Vectorized |
|--|------|-----------|
| **Speed** | Slow (Python overhead) | Very fast (C / SIMD / GPU) |
| **Code length** | Many lines | One or two lines |
| **Readability** | Hard to follow | Matches the math directly |
| **Scalability** | Breaks down at scale | Handles millions of examples |
| **GPU compatible** | No | Yes |

---

## When Loops Are Still Okay

Vectorization is not always the answer. Loops are fine when:

- You're iterating over **epochs** or **training steps** (the number of repeats is small, e.g., 1,000)
- You're processing **files, directories**, or **database records** — not numerical arrays
- The dataset is tiny (fewer than a few hundred examples) and speed doesn't matter
- The logic is complex and cannot easily be expressed as a matrix operation

The rule of thumb: **if you're looping over individual numbers in an array, vectorize it. If you're looping over steps or files, a loop is fine.**

---

## Quick Concept Summary

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| **Vectorization** | Replacing loops with array operations | 100–1000x speed improvement |
| **Vector** | 1D array of numbers | Represents one example's features or model weights |
| **Matrix** | 2D array of numbers | Represents the full dataset |
| **Dot product** | Multiply + sum two vectors | Core of every prediction in ML |
| **NumPy** | Python library for array math | Makes vectorization simple |
| **Broadcasting** | Operating on arrays of different shapes | Removes loops for bias addition and scaling |
| **SIMD** | CPU instruction for parallel math | Why NumPy is fast on CPU |
| **GPU** | Thousands of cores doing math in parallel | Why deep learning is fast at scale |
