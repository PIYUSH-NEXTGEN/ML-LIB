# Linear Regression

Linear Regression is one of the simplest and most widely used supervised machine learning algorithms. Its job is to **predict a continuous numerical value** by learning a relationship between input and output from past data.

**Real-world examples:**
- Predicting a house price based on its size
- Estimating a person's salary based on years of experience
- Forecasting next month's sales from historical data
- Predicting tomorrow's temperature from today's weather

---

## What Is "Regression"?

Imagine you're plotting points on a graph  X-axis is "years of experience", Y-axis is "salary". After plotting 50 employees, you see the points form a rough diagonal pattern. Linear Regression draws the **single best straight line** through all those points.

Once you have this line, you can predict salary for any new experience value.

---
<img width="595" height="469" alt="image" src="https://github.com/user-attachments/assets/5e16706f-d713-4ac5-ab32-b0873a3adfa2" />

## The Linear Regression Equation

$$
\hat{y} = wx + b
$$

| Symbol | Name | Meaning |
|--------|------|---------|
| $x$ | Input Feature | The data you feed in (e.g., years of experience) |
| $\hat{y}$ | Predicted Output | What the model guesses (e.g., predicted salary) |
| $w$ | Weight (Slope) | How much $y$ changes for each unit increase in $x$ |
| $b$ | Bias (Intercept) | The starting value of $y$ when $x = 0$ |

### Intuition with an Example

Say you have a salary prediction model where:
- $w = 5000$ (salary increases by ₹5000 per year of experience)
- $b = 30000$ (a fresher with 0 years earns ₹30,000 base)

For someone with **4 years of experience**:

$$
\hat{y} = 5000 \times 4 + 30000 = 50000
$$

Predicted salary = ₹50,000 ✅

---

## The Goal — Finding the Best $w$ and $b$

The model doesn't know $w$ and $b$ at the start. It **learns** them from the training data. The goal is to find values of $w$ and $b$ that make predictions as close as possible to the actual values.

---

## Cost Function — Measuring How Wrong the Model Is

To know how well (or badly) the model is performing, we use a **Cost Function**. For Linear Regression, we use **Mean Squared Error (MSE)**:

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$$

| Symbol | Meaning |
|--------|---------|
| $m$ | Total number of training examples |
| $\hat{y}_i$ | Predicted value for the $i$-th example |
| $y_i$ | Actual value for the $i$-th example |
| $(\hat{y}_i - y_i)^2$ | Squared difference (penalises big errors heavily) |

### Why Squared?

We square the error for two reasons:
1. **Always positive** — A prediction of 90 when actual is 100 and a prediction of 110 when actual is 100 are both equally wrong; squaring makes both positive.
2. **Big errors penalised more** — An error of 10 becomes 100, but an error of 20 becomes 400. This pushes the model to avoid large mistakes.

### Why the $\frac{1}{2}$?

It's a mathematical convenience. When we take the derivative (during Gradient Descent), the 2 from the square cancels with $\frac{1}{2}$, making the math cleaner. It doesn't change the result.

### Example

Suppose you have 3 data points:

| Actual $y$ | Predicted $\hat{y}$ | Error | Squared Error |
|-----------|-------------------|-------|---------------|
| 100 | 90 | −10 | 100 |
| 200 | 210 | +10 | 100 |
| 150 | 140 | −10 | 100 |

$$
J = \frac{1}{2 \times 3}(100 + 100 + 100) = \frac{300}{6} = 50
$$

A lower J means better predictions. **The model's job is to minimise J.**

---

## Gradient Descent — How the Model Learns

Gradient Descent is the optimisation algorithm that adjusts $w$ and $b$ step-by-step to reduce the cost.

### The Intuition — Walking Down a Hill

Imagine you're blindfolded on a hilly terrain and want to reach the lowest point (minimum cost). You can't see the whole landscape, but you can feel the slope under your feet.

- If the ground slopes down to the right → take a step right
- If the ground slopes down to the left → take a step left
- Keep stepping in the downhill direction until you reach the bottom

Gradient Descent does exactly this in "cost space".

### The Update Rules

$$
w := w - \alpha \frac{\partial J}{\partial w}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

Where:
- $\alpha$ (alpha) is the **learning rate** — the size of each step
- $\frac{\partial J}{\partial w}$ is the **gradient** (slope) of the cost w.r.t. $w$
- $\frac{\partial J}{\partial b}$ is the **gradient** of the cost w.r.t. $b$

### Learning Rate — Critical Hyperparameter

| Learning Rate | Effect |
|---------------|--------|
| Too small (e.g., 0.0001) | Very slow learning, takes thousands of steps |
| Just right (e.g., 0.01) | Converges smoothly to the minimum |
| Too large (e.g., 10) | Overshoots the minimum, may diverge |

### Step-by-Step Process

```
1. Start with random values of w and b (usually 0 or small random numbers)
         ↓
2. Make predictions using ŷ = wx + b
         ↓
3. Calculate the cost J(w, b) using MSE
         ↓
4. Compute the gradients ∂J/∂w and ∂J/∂b
         ↓
5. Update w and b using the update rules
         ↓
6. Repeat from step 2 until cost stops decreasing
```

### Example of One Update Step

Suppose:
- Current $w = 2.0$, $b = 1.0$
- Learning rate $\alpha = 0.1$
- After computing gradients: 

$$
\frac{\partial J}{\partial w} = 0.5, \frac{\partial J}{\partial b} = 0.3 
$$

Updated values:

$$
w = 2.0 - 0.1 \times 0.5 = 1.95
$$

$$
b = 1.0 - 0.1 \times 0.3 = 0.97
$$

The model has taken one small step closer to the optimal line.

---

## Types of Linear Regression

### Simple Linear Regression — One Feature

$$
\hat{y} = wx + b
$$

**Example:** Predicting salary from years of experience alone.

```
Experience (x)  →  [Model: ŷ = 5000x + 30000]  →  Salary (ŷ)
```

### Multiple Linear Regression — Many Features

$$
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

**Example:** Predicting house price from size, number of rooms, location score, and age.

| Feature | Symbol | Weight |
|---------|--------|--------|
| Size (sq ft) | $x_1$ | $w_1 = 300$ |
| Number of rooms | $x_2$ | $w_2 = 5000$ |
| Location score | $x_3$ | $w_3 = 10000$ |
| Age of house | $x_4$ | $w_4 = -200$ |

Prediction for a house (1200 sq ft, 3 rooms, location 8, 10 years old):
$$
\hat{y} = 300(1200) + 5000(3) + 10000(8) + (-200)(10) + b
$$

---

## When to Use Linear Regression

**Use it when:**
- The target variable is **numerical** (not categories like "yes/no")
- There is an approximately **linear relationship** between inputs and output
- You need a **simple, interpretable** model (weights tell you the exact impact of each feature)
- You're working with **small to medium datasets** and want fast training

**Don't use it when:**
- The relationship is clearly non-linear (e.g., exponential growth)
- You're classifying categories (use Logistic Regression instead)
- There are many irrelevant or highly correlated features (need feature selection first)

---

## Full Training Flow

```
Training Data (x, y pairs)
           ↓
  Initialize w = 0, b = 0
           ↓
  ┌─────────────────────────────┐
  │  Make Predictions: ŷ = wx+b │
  │           ↓                 │
  │  Calculate Cost (MSE)       │
  │           ↓                 │
  │  Compute Gradients          │
  │           ↓                 │
  │  Update w and b             │
  └─────────────────────────────┘
           ↓
  Repeat until cost is minimum
           ↓
  Final w and b = Trained Model
           ↓
  New x → Predict ŷ
```

---

## Key Assumptions of Linear Regression

Linear Regression works best when these conditions hold:

1. **Linearity** — The relationship between $x$ and $y$ is actually linear
2. **Independence** — Each training example is independent of others
3. **Homoscedasticity** — The spread of errors is roughly constant across all values of $x$
4. **No multicollinearity** — In multiple regression, features shouldn't be too correlated with each other

---

## Common Pitfalls and How to Handle Them

| Problem | Symptom | Fix |
|---------|---------|-----|
| **Underfitting** | High cost on both train and test data | Use more features, reduce regularisation |
| **Overfitting** | Low train cost but high test cost | Use regularisation (Ridge/Lasso) |
| **Feature scale mismatch** | Gradient descent is slow/unstable | Apply feature normalisation (e.g., min-max scaling) |
| **Outliers** | Model skewed by extreme values | Remove or cap outliers before training |

---

## Quick Concept Summary

| Concept | What It Is | Simple Analogy |
|---------|-----------|----------------|
| $w$ (weight) | How steep the line is | Gas pedal — more $w$, bigger jumps in $\hat{y}$ |
| $b$ (bias) | Where the line starts | Baseline salary even with zero experience |
| Cost $J$ | How wrong the model is | Score on a test — lower is better |
| Gradient Descent | How the model improves | Walking downhill step by step |
| Learning Rate $\alpha$ | How big each step is | Stride length while walking |
| Convergence | When cost stops decreasing | Reaching the bottom of the hill |
