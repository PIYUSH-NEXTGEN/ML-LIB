# Lasso Regression 

Lasso stands for **Least Absolute Shrinkage and Selection Operator**. It is a linear regression model with L1 regularization built directly into the cost function.

If you have already gone through the regularization notes, you know that regularization adds a penalty term to the cost function to prevent overfitting by discouraging large weights. Lasso does exactly this, but with one very special property that sets it apart from Ridge regression: it can push weights all the way to **exactly zero**, effectively removing features from the model entirely.

This makes Lasso not just a regularization technique but also an automatic **feature selection** tool. You feed it 100 features, it trains, and it decides on its own which ones actually matter and which ones should be completely ignored.

---

## Why Lasso Exists

Consider a situation where you have a dataset with 200 features but only 50 of them genuinely affect the output. The other 150 are noise — irrelevant columns that were collected but add nothing useful.

Ordinary linear regression will assign some nonzero weight to all 200 features. It tries to use everything, even the noise, which leads to overfitting. Ridge regression will shrink all weights toward zero but still keeps all 200 features active with small nonzero values. Lasso is the one that actually zeros out the 150 irrelevant features and builds a clean model with only the 50 that matter.

In high-dimensional data — medical genomics, text data, sensor arrays — where you might have thousands of features and most are irrelevant, Lasso is often the go-to starting point.

---

## The Lasso Cost Function

The cost function for Lasso is the standard MSE plus an L1 penalty:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{m} \sum_{j=1}^{n} |w_j|
$$

| Part | What It Does |
|------|-------------|
| $\frac{1}{2m} \sum (\hat{y} - y)^2$ | Original MSE — minimise prediction error |
| $\frac{\lambda}{m} \sum \|w_j\|$ | L1 penalty — sum of absolute values of weights |
| $\lambda$ | Controls how hard the penalty is applied |

The only difference from Ridge (L2) is in the penalty term:

| | Penalty Term | Effect |
|--|-------------|--------|
| Ridge | $\sum w_j^2$ | Shrinks weights toward zero, never exactly zero |
| Lasso | $\sum \|w_j\|$ | Shrinks weights toward zero, can reach exactly zero |

That single difference — absolute value instead of square — produces dramatically different behaviour.

---

## Why Absolute Value Causes Sparsity

This is the most important conceptual point in Lasso and it is worth understanding properly.

With L2 (Ridge), the penalty for a weight $w$ is $w^2$. The gradient of $w^2$ is $2w$, which gets smaller as $w$ approaches zero. So as Ridge pushes a weight toward zero, the force pushing it gets weaker and weaker. The weight asymptotically approaches zero but mathematically never quite reaches it.

With L1 (Lasso), the penalty for a weight $w$ is $|w|$. The gradient of $|w|$ is either $+1$ or $-1$ regardless of the size of $w$. The pushing force is constant. It does not weaken as $w$ approaches zero. This constant pressure is what drives weights all the way to exactly zero and keeps them there.

Geometrically, this shows up in how the constraint regions look:

```
L2 (Ridge) constraint:          L1 (Lasso) constraint:
      w2                               w2
       |                               |
       |   ( )                         |    /\
       |  (   )                        |   /  \
       | (     )                       |  /    \
       |  (   )                        |  \    /
       |   ( )                         |   \  /
       |________________________ w1    |    \/
                                       |________________________ w1

Circle — no corners                 Diamond — has corners at axes

Solution unlikely to land            Solution often lands exactly
exactly on an axis                   on a corner (w = 0 on one axis)
```

The diamond shape of the L1 constraint has sharp corners sitting directly on the axes. The optimal solution — where the cost function contours first touch the constraint region — is much more likely to land on one of these corners, which correspond to exactly zero weights for some features.

---

## The Role of Lambda

Lambda ($\lambda$) controls how aggressively Lasso applies the penalty.

**$\lambda = 0$:** No penalty at all. This is just ordinary linear regression. All features get nonzero weights.

**Small $\lambda$:** Light regularization. Most features kept. Only the truly irrelevant ones get zeroed out.

**Large $\lambda$:** Heavy regularization. More and more features get zeroed out. The model becomes increasingly sparse.

**Very large $\lambda$:** Essentially all weights get pushed to zero. The model predicts the mean of the training data for every input. Severe underfitting.

```
Increasing lambda:

Lambda = 0        Lambda = 0.1      Lambda = 1        Lambda = 100
All 10 features   8 features        5 features        0 features
active            active            active            active

Full model        Mild sparsity     Good sparsity     Underfitting
```

Choosing the right $\lambda$ is done through cross-validation, not guessing.

---

## Gradient Descent for Lasso

The L1 penalty introduces a complication. The absolute value function $|w|$ is not differentiable at $w = 0$ — it has a sharp corner there, so the gradient is undefined at that exact point.

To handle this, Lasso uses the **subgradient** at zero (a generalisation of gradients for non-smooth functions) or a technique called **coordinate descent**, which optimises one weight at a time while holding the others fixed.

The subgradient of $|w|$ is:

$$
\frac{\partial |w|}{\partial w} =
\begin{cases}
+1 & \text{if } w > 0 \\
-1 & \text{if } w < 0 \\
\text{any value in } [-1, +1] & \text{if } w = 0
\end{cases}
$$

In practice, sklearn handles all of this internally using coordinate descent. You do not need to implement the gradient manually.

### The Update Rule (Conceptually)

For each weight $w_j$, the update in coordinate descent involves a **soft thresholding** operation:

$$
w_j = \text{sign}(\rho_j) \cdot \max(|\rho_j| - \lambda, 0)
$$

Where $\rho_j$ is the partial residual — what the weight would be if there were no penalty.

The key behaviour here: if the partial residual is smaller than $\lambda$, the max clips it to exactly zero. This is the mathematical mechanism that produces exactly-zero weights.

---

## Lasso as Feature Selection

Because Lasso zeros out irrelevant weights, you can treat the trained model's coefficients as a feature ranking:

```
Feature          Coefficient
size             0.42
num_rooms        0.18
location_score   0.31
age              0.00   <-- zeroed out, irrelevant
distance_metro   0.00   <-- zeroed out, irrelevant
floor_number     0.09
garden_area      0.00   <-- zeroed out, irrelevant
```

After training, you simply keep the features with nonzero coefficients and discard the rest. This is embedded feature selection — happening automatically as part of model training, not as a separate step.

This is particularly useful when:

- You have far more features than training examples
- You suspect only a small number of features are truly relevant
- You want an interpretable model that only uses the most important features
- You want to understand which inputs are driving predictions

---

## Lasso vs Ridge vs No Regularization

| | No Regularization | Ridge (L2) | Lasso (L1) |
|--|------------------|-----------|-----------|
| Penalty | None | $\sum w_j^2$ | $\sum \|w_j\|$ |
| Shrinks weights | No | Yes, toward zero | Yes, toward zero |
| Can zero out weights | No | No | Yes |
| Feature selection | No | No | Yes (automatic) |
| Best when | Small dataset, few features | Many correlated features | Many irrelevant features |
| Sparsity | None | None | Yes |
| Interpretability | Low (overfits) | Moderate | High (sparse model) |

---

## Elastic Net — The Best of Both

There is a natural follow-up question: what if some features are correlated with each other and some are irrelevant? Ridge handles correlated features better, Lasso handles irrelevant ones better. What do you pick?

The answer is **Elastic Net**, which combines both penalties:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda_1}{m} \sum_{j=1}^{n} |w_j| + \frac{\lambda_2}{2m} \sum_{j=1}^{n} w_j^2
$$

The L1 part handles sparsity and feature selection. The L2 part handles correlated features by grouping them together rather than arbitrarily picking one and zeroing the others.

Elastic Net is the most practical choice when you are not sure whether Ridge or Lasso is more appropriate for your data.

---

## When Lasso Struggles

**Correlated features:** When two features are highly correlated (e.g. house size in square feet and house size in square metres), Lasso tends to pick one arbitrarily and zero out the other. Both carry the same information, but Lasso does not know that — it just picks one. Ridge handles this more gracefully by distributing the weight across both.

**More features than examples:** When you have more columns than rows ($n > m$), Lasso can select at most $m$ features. Beyond that it becomes unpredictable. Elastic Net is more stable in this setting.

**Non-linear relationships:** Lasso is still a linear model. It cannot capture curves or interactions without manual feature engineering first.

---

## Implementation in Code

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling is important for Lasso
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train Lasso with a chosen lambda (alpha in sklearn)
model = Lasso(alpha=0.1)
model.fit(X_train_scaled, y_train)

# See which features survived (nonzero coefficients)
coefficients = pd.Series(model.coef_, index=X.columns)
print(coefficients[coefficients != 0])

# Find best lambda automatically using cross-validation
lasso_cv = LassoCV(alphas=[0.001, 0.01, 0.1, 1, 10], cv=5)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Best lambda: {lasso_cv.alpha_}")
```

Note that sklearn calls the regularization parameter `alpha`, not `lambda`. They mean the same thing.

---

## The Full Picture

```
Dataset with many features (some relevant, many not)
                |
                v
        Apply Lasso regression
        (MSE cost + L1 penalty)
                |
                v
        During training:
        Irrelevant feature weights  →  pushed to exactly zero
        Relevant feature weights    →  shrunk but kept nonzero
                |
                v
        Trained model is sparse:
        Only a subset of features have nonzero weights
                |
                v
        Two outputs in one step:
        1. A linear model that generalises well (regularized)
        2. A list of the features that actually matter (selected)
```

---

## Quick Concept Summary

| Concept | What It Means |
|---------|--------------|
| Lasso | Linear regression with L1 regularization |
| L1 penalty | Sum of absolute values of weights added to the cost |
| Sparsity | Most weights become exactly zero after training |
| Feature selection | Lasso automatically identifies and removes irrelevant features |
| Lambda | Controls penalty strength — higher means more features zeroed out |
| Subgradient | Generalisation of gradient used at the non-differentiable point $w = 0$ |
| Soft thresholding | The operation that clips small weights to exactly zero |
| Coordinate descent | Optimisation method used for Lasso instead of standard gradient descent |
| Ridge vs Lasso | Ridge shrinks weights, Lasso zeros them out |
| Elastic Net | Combines L1 and L2 penalties for the best of both |
