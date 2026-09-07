# Ridge Regression

Ridge Regression is a linear regression model with L2 regularization added to the cost function. It is one of the oldest and most reliable fixes for overfitting in linear models.

The core idea is straightforward. When a linear regression model overfits, it is usually because some weights have grown very large. Large weights mean the model is reacting aggressively to small changes in input, chasing the noise in training data rather than the real signal. Ridge adds a penalty to the cost function that discourages weights from getting large in the first place, forcing the model to find a simpler explanation of the data.

Unlike Lasso, Ridge never pushes weights all the way to zero. It shrinks all of them toward zero but keeps every feature in the model with some nonzero contribution. This makes Ridge the right choice when you believe most of your features are genuinely useful, just perhaps overweighted.

---

## Why Ridge Exists

Ordinary linear regression minimises the sum of squared errors and nothing else. Given enough features relative to the amount of training data, it will find weights that fit the training data almost perfectly — including the noise. The result is a model that performs brilliantly on training data and poorly on anything new.

This problem gets worse as you add more features. With 50 features and 60 training examples, the model has nearly as many parameters as it has data points. It can almost exactly interpolate the training set. The weights become huge and erratic, shaped entirely by the noise in those 60 examples.

Ridge solves this by saying: minimise prediction error, yes — but also keep the weights small. These two objectives pull against each other, and the tension between them is what produces a model that generalises.

---

## The Ridge Cost Function

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2
$$

| Part | What It Does |
|------|-------------|
| $\frac{1}{2m} \sum (\hat{y} - y)^2$ | Standard MSE — fit the training data |
| $\frac{\lambda}{2m} \sum w_j^2$ | L2 penalty — penalise large weights |
| $\lambda$ | Regularization strength — controls the tradeoff |

The bias $b$ is not included in the penalty term. This is standard practice across all regularization methods. The bias simply shifts the prediction up or down and does not contribute to overfitting the way weights do.

The $\frac{1}{2}$ in the penalty is a mathematical convenience. When you differentiate the penalty during gradient descent, the exponent 2 comes down and the $\frac{1}{2}$ cancels it, keeping the gradient formula clean.

---

## What the L2 Penalty Actually Does

The penalty term $\sum w_j^2$ grows quadratically with the size of each weight. A weight of 10 contributes 100 to the penalty. A weight of 20 contributes 400. The model is punished disproportionately for large weights, so it strongly prefers keeping them small.

During training, the model is simultaneously trying to:

- Reduce the MSE by fitting the data better
- Reduce the penalty by keeping weights small

A weight only grows large if doing so meaningfully reduces the prediction error. If a feature is only weakly related to the output, its weight stays near zero because the penalty cost of growing it outweighs the small improvement in fit.

This is what prevents overfitting. The model cannot afford to assign large weights to noisy or marginally useful features.

---

## Gradient Descent for Ridge

The bias update is identical to ordinary linear regression:

$$
b := b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})
$$

The weight update gains an extra term from the penalty:

$$
w_j := w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} w_j \right]
$$

Rearranging to make the effect visible:

$$
w_j := w_j \left(1 - \alpha \frac{\lambda}{m}\right) - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}
$$

The factor $\left(1 - \alpha \frac{\lambda}{m}\right)$ is slightly less than 1. This means every weight is multiplied by a number just below 1 on every single gradient descent step, before the usual gradient update is applied. Each iteration, all weights are gently pulled toward zero. This continuous shrinkage is exactly what Ridge regularization is doing under the hood.

---

## Why Ridge Never Zeros Out Weights

This is the fundamental difference between Ridge and Lasso, and it comes down to geometry.

The L2 penalty is $w^2$. Its gradient is $2w$. As a weight approaches zero, the gradient approaches zero too. The force pulling the weight toward zero gets weaker and weaker the closer you get. The weight asymptotically approaches zero but the shrinkage force fades before it ever reaches it.

Geometrically, the L2 constraint region is a **circle** (in 2D) or a sphere (in higher dimensions). It has no corners or edges. When the cost function contours touch the constraint region, they are overwhelmingly likely to touch at a smooth point on the circle's surface — a point where both weights are nonzero.

```
Ridge (L2) constraint region:          Lasso (L1) constraint region:

        w2                                      w2
         |                                       |
         |    ( )                                |    /\
         |   (   )   <-- circle                 |   /  \  <-- diamond
         |  (     )                             |  /    \
         |   (   )                              |  \    /
         |    ( )                               |   \  /
         |                                      |    \/
         |___________________ w1               |___________________ w1

Optimal solution lands on the             Optimal solution often lands
smooth curve. Both w1 and w2              exactly on a corner where
remain nonzero.                           one weight is exactly zero.
```

The circle has no corners sitting on the axes. The diamond does. That geometric difference is the entire reason Lasso produces sparse models and Ridge does not.

---

## The Role of Lambda

Lambda ($\lambda$) is the single hyperparameter that controls how strongly Ridge penalises large weights.

**$\lambda = 0$:** The penalty term vanishes completely. This is identical to ordinary linear regression with no regularization. All the overfitting problems come back.

**Small $\lambda$ (e.g. 0.001):** Very light regularization. Weights are barely constrained. The model fits the training data closely with only mild shrinkage.

**Moderate $\lambda$ (e.g. 1.0):** A meaningful tradeoff between fitting the data and keeping weights small. Usually where the best generalisation lives.

**Large $\lambda$ (e.g. 1000):** Very strong regularization. All weights are pushed very close to zero. The model becomes almost constant, predicting near the mean of the training output for every input. Underfitting.

```
Increasing lambda  →

Lambda = 0         Lambda = 0.01      Lambda = 1        Lambda = 10000
Large weights      Slightly smaller   Moderate weights  Near-zero weights
Overfitting        weights            Good fit          Underfitting
Low train error    Low train error    Low train error   High train error
High test error    Lower test error   Low test error    High test error
```

You find the right lambda using cross-validation, not intuition.

---

## Ridge Handles Correlated Features Well

One of Ridge's most important practical advantages over Lasso is how it handles correlated features, also called multicollinearity.

Suppose you have two features that carry almost identical information — for example, house size in square feet and house size in square metres. They are perfectly correlated. Ordinary linear regression becomes unstable because there are infinitely many combinations of weights for these two features that produce the same predictions. Any small change in training data shifts the weights wildly.

Lasso deals with this by picking one feature arbitrarily and zeroing out the other. You lose information and the choice is unpredictable.

Ridge deals with this by distributing the weight evenly across both correlated features. If feature A and feature B carry the same information, Ridge assigns them similar weights. The model stays stable and neither feature is discarded.

```
Two correlated features — how each algorithm handles them:

                Feature A weight    Feature B weight
Linear Reg      +500               -490    (unstable, large and opposite)
Lasso           +0.8               0.0     (picks one, zeros the other)
Ridge           +0.4               +0.4    (spreads weight evenly)
```

This stability under multicollinearity is why Ridge is often the first regularization technique to try when features are correlated.

---

## Ridge Has a Closed Form Solution

This is something Lasso does not have. Because the L2 penalty is smooth and differentiable everywhere, Ridge has an exact mathematical solution that can be computed directly without iterating gradient descent:

$$
w = (X^T X + \lambda I)^{-1} X^T y
$$

Where $I$ is the identity matrix. The $\lambda I$ term is added before inverting, which prevents the matrix $X^T X$ from being singular (non-invertible) — a problem that occurs when features are correlated or when there are more features than examples.

This is why Ridge was historically important before gradient descent-based methods dominated. For small to medium datasets, you can compute the exact optimal weights in one step. In practice, sklearn uses this closed-form solution for Ridge when the dataset is small enough.

---

## Ridge vs Lasso vs No Regularization

| | No Regularization | Ridge (L2) | Lasso (L1) |
|--|------------------|-----------|-----------|
| Penalty | None | $\sum w_j^2$ | $\sum \|w_j\|$ |
| Gradient at zero | Not applicable | Approaches zero | Constant |
| Zeros out weights | No | No | Yes |
| Feature selection | No | No | Yes |
| Handles correlated features | Poorly | Well | Poorly |
| Has closed-form solution | Yes | Yes | No |
| Best when | Few features, little noise | Correlated features, all features useful | Many irrelevant features |

---

## When to Use Ridge

Ridge is the right choice when:

- You believe most or all of your features are genuinely relevant to the output
- Features are correlated with each other and you want the model to use all of them
- You want a stable, well-behaved model that does not arbitrarily discard features
- You are working with a dataset where the number of features is close to the number of examples
- You want a model that is less sensitive to small changes in training data

Ridge is not the right choice when:

- You have many irrelevant features and want automatic feature selection — use Lasso
- You need a sparse model for interpretability or deployment efficiency — use Lasso
- You are unsure whether features are correlated or irrelevant — use Elastic Net

---

## Implementation in Code

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling is important for Ridge
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train Ridge with a fixed lambda (called alpha in sklearn)
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

# Inspect weights — all will be nonzero
coefficients = pd.Series(model.coef_, index=X.columns)
print(coefficients.sort_values(key=abs, ascending=False))

# Find the best lambda automatically using cross-validation
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Best lambda: {ridge_cv.alpha_}")

# Predictions and evaluation
y_pred = ridge_cv.predict(X_test_scaled)
```

Note that sklearn names the regularization parameter `alpha` instead of `lambda` to avoid conflict with Python's `lambda` keyword. They mean exactly the same thing.

---

## The Full Picture

```
Dataset where most features are relevant but model is overfitting
                |
                v
        Apply Ridge Regression
        (MSE cost + L2 penalty)
                |
                v
        During training, on every gradient step:
        Each weight multiplied by (1 - α * λ/m)   <-- gentle shrinkage
        Then adjusted by the gradient              <-- still fits data
                |
                v
        Result:
        All weights kept nonzero but smaller
        No feature is discarded
        Model is more stable and generalises better
                |
                v
        Lambda too small  →  still overfitting, increase lambda
        Lambda too large  →  underfitting, decrease lambda
        Lambda just right →  found via cross-validation
```

---

## Quick Concept Summary

| Concept | What It Means |
|---------|--------------|
| Ridge Regression | Linear regression with L2 regularization |
| L2 penalty | Sum of squared weights added to the cost function |
| Weight shrinkage | All weights pulled toward zero on every gradient step |
| Never zeros weights | L2 gradient weakens near zero, so weights approach but never reach it |
| Lambda | Controls regularization strength — higher means more shrinkage |
| Closed-form solution | Ridge has an exact algebraic answer, unlike Lasso |
| Multicollinearity | When features are correlated — Ridge handles this well by distributing weight |
| Overfitting | What Ridge prevents by penalising large weights |
| Underfitting | What happens when lambda is set too high |
| Elastic Net | Combines Ridge and Lasso when you need both shrinkage and sparsity |
