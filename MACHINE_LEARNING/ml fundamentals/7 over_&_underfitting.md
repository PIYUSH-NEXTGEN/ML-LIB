# Overfitting, Underfitting and Regularization 

When you train a machine learning model, the goal is not just to do well on the training data. The real goal is to do well on **new, unseen data**. A model that memorises the training set but fails on new examples is useless in practice.

This is the bias-variance tradeoff, and it shows up in two failure modes: underfitting and overfitting.

---

## Underfitting

Underfitting happens when the model is **too simple** to capture the real pattern in the data. It performs poorly on both training data and new data because it never learned the relationship properly in the first place.

### What It Looks Like

Imagine trying to predict house prices using only one feature and fitting a straight line through data that actually curves. The line misses almost every point.
The model draws a flat line while the actual pattern curves. No matter how much new data you show it, it will keep getting predictions wrong because the model itself lacks the capacity to represent the truth.

### Why It Happens

Underfitting usually comes from one of these:

- The model is too simple for the problem (degree 1 polynomial on curved data)
- Too few features provided to the model
- Too much regularization applied (explained later)
- Training stopped too early before the model had a chance to learn

### Signs of Underfitting

High training error. High test error. Both are bad and roughly similar in magnitude.

---

## Overfitting

Overfitting is the opposite problem. The model is **too complex** and learns the training data too well, including the noise and random fluctuations that are not part of the real pattern.

The model passes through every training point perfectly. But the curve wiggles wildly between points. When a new house comes in, the model predicts a bizarre price because those wiggles have nothing to do with reality.

### Why It Happens

Overfitting usually comes from:

- Model is too complex (too many features, too high a polynomial degree)
- Training data is too small relative to model complexity
- Training for too many iterations without checking test performance
- No regularization applied

### Signs of Overfitting

Very low training error. High test error. The gap between them is the red flag.
<img width="650" height="387" alt="image" src="https://github.com/user-attachments/assets/b32873a6-2b91-4dc7-a528-6c176a7e6f14" />

---

## A Side by Side View

```
                Training Error      Test Error      Problem
Underfitting        High                High         Too simple
Good fit            Low                 Low          Just right
Overfitting         Very Low            High         Too complex
```

The goal is always that middle row: low error on training data and low error on new data. Getting there requires tuning model complexity and using regularization.

---

## Addressing Underfitting

If your model is underfitting, the solution is to give it more capacity or more information.

**Add more features.** If you are predicting salary using only years of experience, adding education level, city, and job role gives the model more to work with.

**Increase polynomial degree.** If a straight line is too simple, try degree 2 or 3.

**Reduce regularization strength.** If you have applied too much regularization, it is forcing the model to stay too simple. Reducing it gives the model more freedom to learn.

**Train longer.** Sometimes the model simply needs more iterations of gradient descent to find the right parameters.

---

## Addressing Overfitting

Overfitting is the more common and more dangerous problem in practice. There are four main ways to fix it.

### 1. Get More Training Data

This is almost always the best fix when available. A more complex model trained on thousands of examples will generalise far better than the same model trained on fifty examples, because there is less room for it to chase noise.

```
Small dataset           Large dataset
  *  *                  * * * *  * *
 *    *          vs    * ** *  * *  *
  * *                  * * * **  * *
  
Model memorises         Model learns the
individual points       actual pattern
```

### 2. Reduce Model Complexity

Use fewer features or a lower polynomial degree. If you have 50 features and only 100 training examples, the model has almost as many parameters as it has data points. It will overfit badly.

Manually remove features that are irrelevant or redundant. Or use feature selection techniques to keep only the most informative ones.

### 3. Regularization

Regularization is a technique that constrains the model's weights to prevent them from becoming too large. Large weights are often a sign of overfitting because they mean the model is reacting very aggressively to small changes in input.

Regularization adds a penalty to the cost function for large weights, forcing the model to keep them small unless there is strong evidence in the data to justify a large value. This will be covered in detail below.

### 4. Dropout (for Neural Networks)

During training, randomly ignore a fraction of neurons on each pass. This prevents any single neuron from becoming too dominant and forces the network to learn more robust, distributed representations. This is specific to neural networks and not covered in depth here.

---

## Regularization 

The core insight behind regularization is simple. Overfitting happens when the model has large weights that react strongly to noise in the training data. If you penalise large weights, the model is forced to find a simpler explanation of the data.

Think of it like this. Without regularization, the model is free to use any weights it wants to minimise training error. With regularization, it has to balance two objectives at once: minimise the prediction error AND keep the weights small. This forces it toward simpler solutions that tend to generalise better.

---

## Cost Function with Regularization

Regularization works by adding a **penalty term** to the cost function.

### The Regularized Cost Function

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

The first term is the original MSE cost. The second term is new — it is the sum of all squared weights, scaled by $\frac{\lambda}{2m}$.

| Part | What It Does |
|------|-------------|
| $\frac{1}{2m} \sum (\hat{y} - y)^2$ | Original cost — minimise prediction error |
| $\frac{\lambda}{2m} \sum w_j^2$ | Regularization penalty — keep weights small |
| $\lambda$ | Regularization parameter — controls the tradeoff |

Notice that the bias $b$ is **not regularized**. This is standard practice. The bias shifts the prediction up or down but does not contribute to overfitting the same way weights do.

### The Role of Lambda ($\lambda$)

$\lambda$ is the most important hyperparameter in regularization. It controls how hard the model is pushed toward small weights.

**$\lambda = 0$:** No regularization at all. The model minimises training error freely. Risk of overfitting.

**Small $\lambda$ (e.g., 0.01):** Light regularization. Model still mostly fits the data but weights are gently discouraged from being large.

**Large $\lambda$ (e.g., 1000):** Heavy regularization. The model is forced to make all weights near zero. Predictions become nearly constant. Risk of underfitting.

**Just right:** A value of $\lambda$ that keeps weights reasonably small while still allowing the model to fit the real pattern. Found by trying different values and checking validation performance.

```
Lambda too small         Lambda just right        Lambda too large
(overfitting)            (good fit)               (underfitting)

  wiggly curve           smooth curve              flat line

  low train error        low train error           high train error
  high test error        low test error            high test error
```

This is called **L2 regularization** or **Ridge regularization** because it penalises the square of the weights.

---

## Regularized Linear Regression

For linear regression, the prediction is:

$$
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

The regularized cost function is:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

### Gradient Descent with Regularization

The update for bias $b$ stays the same:

$$
b := b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

The update for each weight $w_j$ gains an extra regularization term:

$$
w_j := w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right]
$$

This can be rewritten to make the effect clearer:

$$
w_j := w_j \left(1 - \alpha \frac{\lambda}{m}\right) - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}
$$

The factor $\left(1 - \alpha \frac{\lambda}{m}\right)$ is slightly less than 1. So before applying the usual gradient step, every weight is **shrunk** by a small amount. This shrinkage is what prevents weights from growing large. It happens on every iteration, every step, continuously throughout training.

### Vectorized Implementation

```python
# Prediction
y_hat = np.dot(X, w) + b

# Gradients
errors = y_hat - y
grad_w = (np.dot(X.T, errors) + lambda_ * w) / m
grad_b = np.sum(errors) / m

# Update
w = w - alpha * grad_w
b = b - alpha * grad_b
```

The only change from unregularized gradient descent is `+ lambda_ * w` added to the weight gradient.

---

## Regularized Logistic Regression

For logistic regression, the prediction uses the sigmoid:

$$
\hat{y} = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}
$$

The regularized cost function is:

$$
J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

Same structure as before. Original log loss cost on the left. L2 regularization penalty on the right.

### Gradient Descent with Regularization

The bias update stays identical:

$$
b := b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

The weight update gains the same regularization term:

$$
w_j := w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right]
$$

The gradient formulas look identical to regularized linear regression. The only difference is that $\hat{y}$ comes from the sigmoid in logistic regression.

### Vectorized Implementation

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Prediction
z     = np.dot(X, w) + b
y_hat = sigmoid(z)

# Gradients
errors = y_hat - y
grad_w = (np.dot(X.T, errors) + lambda_ * w) / m
grad_b = np.sum(errors) / m

# Update
w = w - alpha * grad_w
b = b - alpha * grad_b
```

Swap `sigmoid(z)` for `z` and this becomes regularized linear regression. The regularization part is identical in both.

---

## L1 vs L2 Regularization

The regularization discussed so far (squaring the weights) is called L2. There is another common type called L1.

| | L1 (Lasso) | L2 (Ridge) |
|--|-----------|-----------|
| Penalty term | $\frac{\lambda}{m} \sum \|w_j\|$ | $\frac{\lambda}{2m} \sum w_j^2$ |
| Effect on weights | Pushes some weights to exactly zero | Shrinks all weights toward zero |
| Result | Sparse model (automatic feature selection) | All features kept but smaller |
| Use when | You suspect many features are irrelevant | You want to keep all features but controlled |

L2 is more commonly used and is what most people mean when they say "regularization" without specifying. L1 is useful when you want the model to automatically ignore irrelevant features by setting their weights to zero.

---

## Choosing Lambda in Practice

You cannot just guess the right value of $\lambda$. The standard approach is:

**Step 1:** Split data into training set, validation set and test set.

**Step 2:** Train the model with several values of $\lambda$, for example: 0, 0.001, 0.01, 0.1, 1, 10, 100.

**Step 3:** For each $\lambda$, evaluate the cost on the validation set (not the test set).

**Step 4:** Pick the $\lambda$ that gives the lowest validation cost.

**Step 5:** Report the final performance on the test set using the chosen $\lambda$.

The test set is only touched once at the very end. Using it to pick $\lambda$ would be data leakage.

---

## Quick Concept Summary

| Concept | What It Means |
|---------|--------------|
| Underfitting | Model too simple, high error on both train and test |
| Overfitting | Model too complex, low train error but high test error |
| Regularization | Adding a weight penalty to the cost function to prevent overfitting |
| $\lambda$ | Controls strength of regularization — higher means more shrinkage |
| L2 regularization | Penalises squared weights, shrinks all weights toward zero |
| L1 regularization | Penalises absolute weights, can zero out irrelevant features entirely |
| Weight shrinkage | Each weight is multiplied by slightly less than 1 on every update |
| Bias not regularized | Standard practice — bias does not contribute to overfitting |
