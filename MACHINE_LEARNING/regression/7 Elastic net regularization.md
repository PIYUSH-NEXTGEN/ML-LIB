# Elastic Net Regularization 

Elastic Net is a regularization technique that combines both L1 (Lasso) and L2 (Ridge) penalties into a single cost function. It was introduced specifically to address the situations where neither Ridge nor Lasso alone gives a satisfying answer.

If you have already gone through the Ridge and Lasso notes, you know that each has a specific weakness. Ridge handles correlated features well but never removes any feature entirely. Lasso removes irrelevant features automatically but behaves unpredictably when features are correlated. Elastic Net sits in the middle, taking the strengths of both and softening the weaknesses of each.

Think of it as a dial between pure Ridge and pure Lasso. You can lean it more toward one or the other depending on what your data needs. At one extreme it becomes Ridge. At the other it becomes Lasso. Everywhere in between it is a blend of both.

---

## Why Neither Ridge Nor Lasso Is Always Enough

Before getting into the formula, it is worth being concrete about exactly when each method struggles.

**When Ridge struggles:** You have 300 features and you suspect that 250 of them are noise — irrelevant columns that were collected but do not actually predict the output. Ridge will shrink all 300 weights but keep every single one of them active. Your model is unnecessarily complex and hard to interpret.

**When Lasso struggles:** You have 10 features and several pairs of them are highly correlated — for example, a customer's total spend and their average order value, which move together. Lasso will arbitrarily pick one from each correlated pair and zero out the other. You lose information, and which one gets kept is sensitive to small changes in the training data.

**When you have both problems at once:** Some features are irrelevant (Lasso's strength needed) and some relevant features are correlated with each other (Ridge's strength needed). This is extremely common in practice — real datasets are messy. Elastic Net is built exactly for this situation.

---

## The Elastic Net Cost Function

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda_1}{m} \sum_{j=1}^{n} |w_j| + \frac{\lambda_2}{2m} \sum_{j=1}^{n} w_j^2
$$

| Part | Type | Effect |
|------|------|--------|
| $\frac{1}{2m} \sum (\hat{y} - y)^2$ | MSE | Fit the training data |
| $\frac{\lambda_1}{m} \sum \|w_j\|$ | L1 penalty | Push irrelevant weights to exactly zero |
| $\frac{\lambda_2}{2m} \sum w_j^2$ | L2 penalty | Shrink correlated weights smoothly |

You now have two regularization parameters instead of one. $\lambda_1$ controls how much Lasso behaviour you get and $\lambda_2$ controls how much Ridge behaviour you get.

### The Mixing Parameter Formulation

In sklearn and many textbooks, Elastic Net is expressed slightly differently using a single overall penalty strength $\lambda$ and a mixing parameter $r$ (called `l1_ratio` in sklearn):

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{m} \left[ r \sum_{j=1}^{n} |w_j| + \frac{(1-r)}{2} \sum_{j=1}^{n} w_j^2 \right]
$$

| Parameter | Meaning | Effect |
|-----------|---------|--------|
| $\lambda$ | Overall regularization strength | Higher means more penalty overall |
| $r$ (l1_ratio) | Mix between L1 and L2 | $r=1$ is pure Lasso, $r=0$ is pure Ridge |

This formulation is more convenient because you can search over one strength and one ratio independently, which is easier to tune than two separate lambda values.

---

## How the Two Penalties Work Together

The L1 and L2 penalties each do their job simultaneously during training.

The L1 part applies constant pressure on every weight regardless of its size. Weights that do not earn their place — features that are not contributing meaningfully to reducing the prediction error — get pushed all the way to zero. The sparsity from L1 keeps the model clean.

The L2 part applies shrinkage proportional to the size of the weight. When two features are correlated and both deserve some weight, the L2 penalty distributes that weight across both of them rather than concentrating it all on one. The stability from L2 keeps the model well-behaved on correlated features.

The geometric constraint region for Elastic Net sits between the circle of Ridge and the diamond of Lasso — it looks like a rounded diamond. It has corners (producing zeros like Lasso) but the edges are curved rather than flat (preventing the instability of pure Lasso on correlated features).

```
Ridge (circle)        Elastic Net              Lasso (diamond)
                      (rounded diamond)

      w2                    w2                      w2
       |                     |                       |
       |  ( )                |   / \                 |   /\
       | (   )               |  /   \                |  /  \
       |  ( )                |  \   /                |  \  /
       |                     |   \ /                 |   \/
       |___________ w1       |___________ w1         |___________ w1

No corners            Corners exist but          Sharp corners
No sparsity           edges are curved            Full sparsity
                      Partial sparsity
```

The rounded corners mean the optimal solution can still land on an axis (zeroing a weight) but is less likely to do so than with pure Lasso. You get some sparsity, just not as aggressively as Lasso alone.

---

## The Effect of Each Parameter

### Effect of Lambda (Overall Strength)

This works the same way as in Ridge and Lasso individually.

| Lambda | Effect |
|--------|--------|
| Near zero | Very weak regularization, close to ordinary linear regression |
| Small | Light shrinkage and mild sparsity |
| Moderate | Meaningful shrinkage, some features zeroed out |
| Large | Heavy shrinkage, many features zeroed out |
| Very large | Almost all weights near zero, underfitting |

### Effect of l1_ratio (The Mix)

This is what makes Elastic Net unique.

| l1_ratio | Behaviour |
|----------|-----------|
| 0.0 | Pure Ridge — no sparsity, handles correlated features best |
| 0.1 to 0.3 | Mostly Ridge with a hint of sparsity |
| 0.5 | Equal mix of L1 and L2 |
| 0.7 to 0.9 | Mostly Lasso with Ridge stabilising correlated features |
| 1.0 | Pure Lasso — full sparsity, may arbitrarily drop correlated features |

In practice, values between 0.1 and 0.9 are where Elastic Net earns its place. The exact value is found through cross-validation.

---

## Gradient Descent for Elastic Net

The update rule for each weight combines both penalties:

$$
w_j := w_j - \alpha \left[ \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)} + \frac{\lambda_2}{m} w_j + \frac{\lambda_1}{m} \cdot \text{sign}(w_j) \right]
$$

Breaking this down:

| Term | Comes From | Does |
|------|-----------|------|
| $\frac{1}{m} \sum (\hat{y} - y) x_j$ | MSE gradient | Updates weight to fit data better |
| $\frac{\lambda_2}{m} w_j$ | L2 gradient | Shrinks weight proportionally |
| $\frac{\lambda_1}{m} \cdot \text{sign}(w_j)$ | L1 subgradient | Applies constant push toward zero |

Like Lasso, Elastic Net uses coordinate descent in practice because of the non-differentiability of the L1 term at zero. The soft thresholding from Lasso is applied first, then the Ridge shrinkage on top of that.

---

## When Features Are Grouped and Correlated

One particularly useful property of Elastic Net is called the **grouping effect**. When several features are strongly correlated with each other, Elastic Net tends to assign them similar weight values and either keeps the whole group or zeros out the whole group together.

Lasso by contrast picks one from the group arbitrarily and zeros the rest. Which one gets picked can change with a slightly different training sample. Elastic Net is consistent.

**Example — predicting house price with correlated features:**

```
Features: [total_area, living_area, floor_area]
(These three are highly correlated with each other)

Lasso result:
  total_area    0.45
  living_area   0.00   <-- dropped arbitrarily
  floor_area    0.00   <-- dropped arbitrarily

Elastic Net result (l1_ratio = 0.5):
  total_area    0.18
  living_area   0.15
  floor_area    0.14
```

Elastic Net recognised these three carry similar information and spread the weight across all of them. The model is more stable and the result makes more intuitive sense — all three area measurements should matter somewhat.

---

## Elastic Net vs Ridge vs Lasso

| | Ridge | Lasso | Elastic Net |
|--|-------|-------|-------------|
| Penalty | L2 only | L1 only | L1 + L2 |
| Zeros out weights | No | Yes | Yes (some) |
| Feature selection | No | Yes | Yes (partial) |
| Handles correlated features | Well | Poorly | Well |
| Grouping effect | No | No | Yes |
| Number of hyperparameters | 1 ($\lambda$) | 1 ($\lambda$) | 2 ($\lambda$ and $r$) |
| Closed-form solution | Yes | No | No |
| Best when | All features relevant, some correlated | Many irrelevant features | Both problems present |

---

## When to Use Elastic Net

Elastic Net is the right choice when:

- You have both irrelevant features and correlated features in the same dataset
- You tried Lasso and found it was unstable or kept dropping useful correlated features
- You tried Ridge and found the model was too complex with too many active features
- The number of features is large relative to the number of training examples
- You want some sparsity but not as aggressive as pure Lasso
- In high-dimensional settings like genomics, text classification, or finance where both problems are common

Elastic Net is not necessarily better than Ridge or Lasso in all situations. When your data clearly fits the profile of one of them, use that one. Elastic Net is the safe middle-ground choice when you are not sure or when both patterns are present.

---

## Implementation in Code

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Always scale before Elastic Net
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train with fixed hyperparameters
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train_scaled, y_train)

# See which features survived
coefficients = pd.Series(model.coef_, index=X.columns)
print("Nonzero features:")
print(coefficients[coefficients != 0].sort_values(key=abs, ascending=False))

print(f"\nFeatures zeroed out: {(coefficients == 0).sum()}")

# Find the best alpha and l1_ratio using cross-validation
l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]

model_cv = ElasticNetCV(
    l1_ratio=l1_ratios,
    alphas=[0.001, 0.01, 0.1, 1, 10],
    cv=5,
    random_state=42
)
model_cv.fit(X_train_scaled, y_train)

print(f"Best alpha: {model_cv.alpha_}")
print(f"Best l1_ratio: {model_cv.l1_ratio_}")

y_pred = model_cv.predict(X_test_scaled)
```

---

## A Practical Decision Guide

When you are choosing between Ridge, Lasso, and Elastic Net, this flow helps:

```
Do you have irrelevant features that should be removed?
        |
       Yes                          No
        |                            |
Do you also have correlated        Use Ridge
features that should be kept?
        |
       Yes                          No
        |                            |
Use Elastic Net                  Use Lasso
(l1_ratio between 0.3 and 0.9)
```

If you are truly unsure and cannot tell from the data, Elastic Net with `l1_ratio=0.5` is a reasonable starting point. Cross-validation will then tune both parameters toward whatever the data actually needs.

---

## The Full Picture

```
Dataset with a mix of irrelevant features and correlated features
                |
                v
        Apply Elastic Net
        (MSE + L1 penalty + L2 penalty)
                |
                v
During training, on every gradient step:

        L1 part     →  constant pressure on all weights
                       irrelevant features eventually reach zero

        L2 part     →  proportional shrinkage on all weights
                       correlated features share weight evenly

                |
                v
        Result:
        Irrelevant features  →  zeroed out (from L1)
        Correlated features  →  kept with shared smaller weights (from L2)
        Clean, stable, sparse model that generalises well
                |
                v
        Tune lambda and l1_ratio via cross-validation
```

---

## Quick Concept Summary

| Concept | What It Means |
|---------|--------------|
| Elastic Net | Regularization combining both L1 and L2 penalties |
| L1 part | Pushes irrelevant weights to exactly zero — sparsity |
| L2 part | Shrinks and stabilises correlated weights — stability |
| Lambda | Overall regularization strength — higher means more penalty |
| l1_ratio | Mix between L1 and L2 — 0 is pure Ridge, 1 is pure Lasso |
| Grouping effect | Correlated features get similar weights instead of one being arbitrarily zeroed |
| Rounded diamond | The geometric constraint shape sitting between Ridge circle and Lasso diamond |
| When to use | Dataset has both irrelevant features and correlated features |
| ElasticNetCV | Sklearn utility that finds the best lambda and l1_ratio via cross-validation |
