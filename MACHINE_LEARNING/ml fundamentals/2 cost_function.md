# Cost Function in Machine Learning 

A **cost function** is a mathematical function that measures how wrong a machine learning model's predictions are across the entire training dataset. It converts all the individual prediction errors into a **single number**.

- Low Cost → Good predictions (model is learning well)
- High Cost → Poor predictions (model needs improvement)

The entire goal of training is to **minimise this number**.

---

## Why Do We Need a Cost Function?

When a model makes predictions, it needs a feedback mechanism :

- *How far off were my predictions?*
- *Am I getting better or worse after each update?*
- *In which direction should I adjust my parameters?*

Without a cost function, the model is essentially guessing blindly. The cost function is the **compass** that guides learning.

### Analogy

Think of a student learning archery. Each arrow lands somewhere on the target. The score tells them how far from the bullseye they are and whether their technique is improving. The cost function is that score for a machine learning model.

---

## Loss Function vs Cost Function 

These two terms are often confused. Here's the exact distinction:

| | Loss Function | Cost Function |
|--|--------------|---------------|
| **Scope** | A single training example | The entire dataset |
| **Purpose** | Measures one prediction's error | Averages all losses into one value |
| **Used for** | Understanding individual errors | Guiding the overall learning process |

### Example

Suppose you have 3 house price predictions:

| Example | Actual $y$ | Predicted $\hat{y}$ | Loss $(\ \hat{y} - y)^2$ |
|---------|-----------|-------------------|--------------------------|
| 1 | 300 | 250 | $(250-300)^2 = 2500$ |
| 2 | 500 | 480 | $(480-500)^2 = 400$ |
| 3 | 200 | 230 | $(230-200)^2 = 900$ |

**Loss** = error for each individual row above.

**Cost** = average of all losses:

$$
J = \frac{1}{2 \times 3}(2500 + 400 + 900) = \frac{3800}{6} \approx 633.3
$$

The cost is the **single number** the model tries to minimise.

---

## Mean Squared Error (MSE)

For Linear Regression, the standard cost function is **Mean Squared Error**:

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$$

| Symbol | Meaning |
|--------|---------|
| $m$ | Total number of training examples |
| $\hat{y}_i$ | Model's predicted value for the $i$-th example |
| $y_i$ | Actual (true) value for the $i$-th example |
| $(\hat{y}_i - y_i)^2$ | Squared error for one example |
| $J(w,b)$ | The final cost — one number summarising all errors |

### Why $\frac{1}{2}$?

The $\frac{1}{2}$ is a **mathematical convenience**, not a conceptual necessity. When we differentiate $J$ during Gradient Descent, the exponent 2 comes down and multiplies the expression. The $\frac{1}{2}$ cancels it out, keeping the gradient formula clean:

$$
\frac{\partial}{\partial w}\left[\frac{1}{2m}(\hat{y}-y)^2\right] = \frac{1}{m}(\hat{y}-y)
$$

Without $\frac{1}{2}$, there'd be a stray 2 in every gradient formula. It doesn't affect which $w$ and $b$ minimise the cost.

---

## Why Square the Error?

This is one of the most important design decisions in the cost function. Here's why squaring is necessary:

### Problem: Errors Can Cancel Out


Prediction A: 310  →  Error = +10
Prediction B: 290  →  Error = -10

Sum of errors = 10 + (-10) = 0
---

## The Cost Function as a Landscape

Here's a powerful way to visualise what the cost function represents.

The model has two parameters: w (weight) and b (bias). For every possible combination of w and b, 
the cost function produces a value. If you plot this in 3D, you get a bowl-shaped surface:

- The top of the bowl = high cost = bad model
- The bottom of the bowl = minimum cost = best possible model

The goal of training is to find the bottom of this bowl.

### Why Is MSE Bowl-Shaped?

Because MSE is a quadratic function of w and b (it involves squaring). 
Quadratic functions produce parabolas in 2D and bowls in 3D — and crucially, 
they have exactly one global minimum. 
This is why Gradient Descent always finds the best solution for Linear Regression.

---

## How the Cost Function Guides Training

The training cycle uses the cost function at every step:

Each iteration, the model asks: "Which direction should I nudge w and b to reduce the cost?"* The gradient of the cost function answers this question.

---

## Cost Function and Gradient Descent 

The cost function and Gradient Descent are two halves of the same learning process:

| | Role |
|--|------|
| **Cost Function** | Defines what "wrong" looks like — the landscape |
| **Gradient Descent** | Navigates the landscape to find the lowest point |

Neither works without the other. The cost function without Gradient Descent just measures error passively. Gradient Descent without a cost function has no landscape to navigate.

### Analogy

Imagine you're trying to find the lowest point in a dark valley:

- The **cost function** is the terrain itself  hills and valleys
- The **gradient** is the slope under your feet at any moment
- Gradient Descent is the strategy: always step in the direction the slope goes down

---

## Common Cost Functions by Problem Type

Different problems need different cost functions. The choice depends on what kind of output the model produces.

### 1. Linear Regression → Mean Squared Error (MSE)

$$
J = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}_i - y_i)^2
$$

Used when predicting **continuous numbers** (prices, temperatures, salaries). Penalises large errors quadratically.

### 2. Binary Classification → Binary Cross-Entropy

$$
J = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]
$$

Used when the output is one of **two classes** (spam/not spam, disease/no disease). Measures how well the predicted *probability* matches the true label.

**Intuition:** If the actual label is 1 and the model predicts 0.99 (very confident and correct), cost is near 0. If the model predicts 0.01 (very confident and wrong), cost is very high.

### 3. Multi-Class Classification → Categorical Cross-Entropy

$$
J = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})
$$

Used when the output can be one of **many classes** (cat/dog/bird, digit 0–9). Generalises binary cross-entropy to $K$ classes.

### Summary Table

| Problem Type | Example | Cost Function |
|--------------|---------|---------------|
| Regression | House price prediction | Mean Squared Error |
| Binary Classification | Email spam detection | Binary Cross-Entropy |
| Multi-Class Classification | Handwritten digit recognition | Categorical Cross-Entropy |

---

## How to Know If Cost Is Improving — The Learning Curve

During training, you plot the cost over iterations. A healthy training run looks like this:

```
Cost J
  │
  │ \
  │  \
  │   \
  │    \___
  │        \____
  │              \________
  └──────────────────────────── Iterations
```

- Cost should **steadily decrease** and eventually flatten out
- If cost is **oscillating wildly** → learning rate is too high
- If cost is **decreasing too slowly** → learning rate is too low
- If cost **increases** at some point → something is likely wrong (e.g., NaN values, too large a learning rate)

---

## Quick Concept Summary

| Concept | What It Does | Simple Analogy |
|---------|-------------|----------------|
| Cost Function | Scores how wrong the model is | Exam score — lower is better |
| Loss | Error on one training example | One wrong answer on a test |
| Cost | Average loss over all examples | Your overall test score |
| MSE | Specific cost function for regression | Measuring how far arrows land from bullseye |
| Squaring errors | Prevents cancellation, penalises big errors | A miss of 10m is 4× worse than 5m |
| Minimum cost | Best possible parameter values | The bottom of the valley |
| Gradient Descent | Algorithm that minimises cost | Walking downhill in the dark |
