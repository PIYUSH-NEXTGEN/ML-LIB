# Logistic Regression 
Say you want to predict whether a tumour is malignant (1) or benign (0) based on its size. If you fit a straight line through this data, the model might predict values like 1.8 or -0.3  numbers that make no sense as probabilities. A probability has to be between 0 and 1.

Worse, if you add a few extreme data points far to the right, the line shifts and your decision threshold shifts with it, misclassifying cases that were previously correct.

Logistic regression fixes this by squashing predictions into the range (0, 1) using a special function the sigmoid.

---
Despite the name, logistic regression is a **classification** algorithm, not a regression one. The name comes from the function it uses internally, not what it predicts.

It predicts the **probability** that a given input belongs to a particular class. Then it converts that probability into a class label using a threshold.

Common examples:
- Email is spam or not spam
- Tumour is malignant or benign
- Customer will churn or stay
- Loan applicant will default or repay

---

## The Sigmoid Function

The core of logistic regression is the **sigmoid function**, which takes any real number and maps it to a value strictly between 0 and 1.

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Where $z$ is the linear combination of inputs:

$$
z = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

So the full prediction is:

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

### What the Sigmoid Looks Like

```
Output
  1.0  |                          __________
       |                     ____/
  0.5  |                ____/
       |           ____/
  0.0  |__________/
       |_________________________________ z
            negative    0    positive
```

- When $z$ is a large positive number, $\sigma(z)$ approaches 1
- When $z$ is a large negative number, $\sigma(z)$ approaches 0
- When $z = 0$, $\sigma(z) = 0.5$ exactly

This output is interpreted as a probability. If $\hat{y} = 0.8$, the model is saying there is an 80% chance the input belongs to class 1.

### A Quick Example

Suppose for a given patient, $z = 2.5$:

$$
\hat{y} = \frac{1}{1 + e^{-2.5}} = \frac{1}{1 + 0.082} \approx 0.924
$$

The model predicts a 92.4% chance the tumour is malignant.

---

## From Probability to Class Label

The model outputs a probability. To get an actual class prediction (0 or 1), you apply a threshold usually 0.5.

$$
\text{Prediction} =
\begin{cases}
1 & \text{if } \hat{y} \geq 0.5 \\
0 & \text{if } \hat{y} < 0.5
\end{cases}
$$

Since $\hat{y} \geq 0.5$ when $z \geq 0$, you can also think of it as:

$$
\text{Predict class 1 if } w_1x_1 + w_2x_2 + \cdots + b \geq 0
$$

The threshold of 0.5 is the default but can be adjusted depending on the problem. In cancer detection you might lower it to 0.3 to catch more true positives, even at the cost of more false alarms.

---

## Decision Boundary

The **decision boundary** is the line (or curve) that separates the region where the model predicts class 0 from the region where it predicts class 1.

It sits exactly where $z = 0$, because that is where $\hat{y} = 0.5$  the point of maximum uncertainty.

### Linear Decision Boundary

With two features $x_1$ and $x_2$:

$$
z = w_1x_1 + w_2x_2 + b = 0
$$

This is the equation of a straight line in the feature space. Everything on one side gets labelled class 1, everything on the other side gets labelled class 0.

**Example:** Suppose the model learns $w_1 = 1$, $w_2 = -2$, $b = 3$.

The decision boundary is:

$$
x_1 - 2x_2 + 3 = 0 \quad \Rightarrow \quad x_1 = 2x_2 - 3
$$

Any point where $x_1 - 2x_2 + 3 \geq 0$ gets predicted as class 1.

### Non-Linear Decision Boundary

If the data cannot be separated by a straight line, you can add polynomial features just like in polynomial regression.

For example, with features $x_1^2$ and $x_2^2$ added:

$$
z = w_1x_1^2 + w_2x_2^2 + b = 0
$$

This produces a circular or elliptical boundary — useful when one class clusters in the middle and the other surrounds it.

```
Non-linear boundary example:

    x2
     |      o o o
     |    o       o
     |   o    *    o       o = class 0
     |    o  * *  o        * = class 1
     |      o o o
     |________________________ x1

The boundary is a circle separating the two classes.
```

The decision boundary itself is a property of the parameters  it gets learned from data, not specified by hand.

---

## Cost Function for Logistic Regression

Here is where logistic regression parts ways with linear regression. You cannot use Mean Squared Error (MSE) as the cost function for logistic regression.

### Why MSE Fails Here

With the sigmoid function, the cost surface becomes **non-convex** it has many local minima. Gradient descent would get stuck and never reliably find the best parameters.

You need a cost function that is convex (bowl-shaped) so gradient descent always converges to the global minimum.

### Log Loss (Binary Cross-Entropy)

The correct cost function for logistic regression is **Log Loss**, also called Binary Cross-Entropy:

For a single training example:

$$
\text{Loss} =
\begin{cases}
-\log(\hat{y}) & \text{if } y = 1 \\
-\log(1 - \hat{y}) & \text{if } y = 0
\end{cases}
$$

This can be written more compactly as:

$$
\text{Loss} = -\left[ y \log(\hat{y}) + (1 - y)\log(1 - \hat{y}) \right]
$$

When $y = 1$ the second term vanishes (since $1 - 1 = 0$), leaving $-\log(\hat{y})$.
When $y = 0$ the first term vanishes, leaving $-\log(1 - \hat{y})$.

### Intuition Behind the Loss

**When actual label is 1:**

$$
\text{Loss} = -\log(\hat{y})
$$

| Predicted $\hat{y}$ | Loss |
|--------------------|------|
| 0.99 (confident, correct) | $-\log(0.99) \approx 0.01$ — nearly zero |
| 0.50 (uncertain) | $-\log(0.50) \approx 0.69$ — moderate |
| 0.01 (confident, wrong) | $-\log(0.01) \approx 4.6$ — very high |

The model is punished heavily when it is confident but wrong.

**When actual label is 0:**

$$
\text{Loss} = -\log(1 - \hat{y})
$$

| Predicted $\hat{y}$ | Loss |
|--------------------|------|
| 0.01 (confident, correct) | $-\log(0.99) \approx 0.01$ — nearly zero |
| 0.50 (uncertain) | $-\log(0.50) \approx 0.69$ — moderate |
| 0.99 (confident, wrong) | $-\log(0.01) \approx 4.6$ — very high |

The pattern is the same. Correct and confident = low loss. Wrong and confident = high loss.

### The Full Cost Function

The cost function averages the loss across all $m$ training examples:

$$
J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

This function is **convex** it has a single global minimum, so gradient descent will always find the best parameters.

---

## Gradient Descent for Logistic Regression

The goal is the same as linear regression: minimise $J(w, b)$ by repeatedly updating the parameters.

### Update Rules

$$
w_j := w_j - \alpha \frac{\partial J}{\partial w_j}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

After computing the partial derivatives of the log loss, the gradients work out to:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

These look identical to the linear regression gradients — and that is not a coincidence. The math works out the same because of how the log loss is defined. The key difference is that $\hat{y}$ here is the **sigmoid** of $z$, not $z$ itself.

### Why This Is Convenient

Even though the cost function changed, the gradient update formulas look the same. If you already understand gradient descent for linear regression, the implementation for logistic regression is almost identical — just swap the prediction step to use the sigmoid.

### Vectorized Implementation

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialise
w = np.zeros(n)
b = 0
alpha = 0.01
iterations = 1000

for _ in range(iterations):

    # Step 1: Predict (sigmoid of linear combination)
    z     = np.dot(X, w) + b
    y_hat = sigmoid(z)

    # Step 2: Compute gradients
    errors = y_hat - y
    grad_w = np.dot(X.T, errors) / m
    grad_b = np.sum(errors) / m

    # Step 3: Update simultaneously
    w = w - alpha * grad_w
    b = b - alpha * grad_b
```

The only line that changed from linear regression is the prediction step — `sigmoid(z)` instead of just `z`.

---

## Putting It All Together

Here is how all the pieces connect:

```
Input features (x)
        |
        v
z = w·x + b          <-- linear combination (same as linear regression)
        |
        v
ŷ = sigmoid(z)        <-- squash into (0, 1) range
        |
        v
Compare ŷ with y      <-- compute log loss
        |
        v
Compute gradients     <-- how should w and b change?
        |
        v
Update w and b        <-- gradient descent step
        |
        v
Repeat until cost is minimum
        |
        v
Decision boundary     <-- where z = 0 in the learned model
        |
        v
Threshold at 0.5      <-- final class prediction (0 or 1)
```

---

## Logistic vs Linear Regression — Key Differences

| | Linear Regression | Logistic Regression |
|--|------------------|---------------------|
| Task | Predict a number | Predict a class |
| Output | Any real number | Probability between 0 and 1 |
| Activation | None (identity) | Sigmoid function |
| Cost function | Mean Squared Error | Log Loss (Binary Cross-Entropy) |
| Cost surface | Convex | Convex (because of log loss) |
| Gradient formulas | Same form | Same form (different ŷ) |
| Decision boundary | Not applicable | Where z = 0 |

---

## Quick Concept Summary

| Concept | What It Is |
|---------|-----------|
| Sigmoid | Squashes any number to (0, 1), outputs a probability |
| $z$ | Linear combination $w \cdot x + b$ before sigmoid |
| $\hat{y}$ | Predicted probability after sigmoid |
| Decision boundary | Where $z = 0$ — the line separating class 0 and class 1 |
| Log loss | Cost function that penalises confident wrong predictions heavily |
| Gradient descent | Identical structure to linear regression, just uses sigmoid predictions |
| Threshold | Converts probability to class label — default is 0.5 |
