# Gradient Boosting Machine (GBM) 

Gradient Boosting is an ensemble learning technique that builds a strong predictive model by combining many weak models — typically shallow decision trees — one at a time, where each new model is specifically trained to fix the mistakes of everything built before it.

Unlike Random Forest, which builds all trees independently in parallel and averages their votes, Gradient Boosting builds trees **sequentially**. Each tree learns from the errors of the previous ensemble. The models are not independent — each one is a direct response to where the current model is going wrong.

The word "boosting" refers to this idea of sequentially improving a weak learner into a strong one. The word "gradient" refers to how errors are defined and minimised — using the same gradient descent logic you already know from linear regression, but applied to the space of models rather than the space of parameters.

---

## The Core Idea — Learning from Mistakes

The best way to understand Gradient Boosting is through a simple analogy.

Imagine you are a student taking practice tests. After your first attempt, you check your answers. Instead of reviewing everything, you focus only on the questions you got wrong. Your second attempt targets those weak spots. Then you check again and focus on whatever remains wrong. You keep iterating, each round specifically addressing your current weaknesses.

Gradient Boosting works exactly this way. Each tree studies the residual errors — the gap between what the current model predicts and what the actual answer is — and tries to predict those errors. The corrected predictions are then passed to the next tree, which again studies whatever errors remain.

```
Actual price of house: ₹80L

Round 1: First tree predicts          ₹60L    (error: +₹20L)
Round 2: Second tree predicts error   ₹15L    (error: +₹5L remaining)
Round 3: Third tree predicts error    ₹4L     (error: +₹1L remaining)
Round 4: Fourth tree predicts error   ₹0.8L   (almost there)
...

Final prediction = 60 + 15 + 4 + 0.8 + ... = close to ₹80L
```

Each tree is small and weak on its own. Together they converge toward the correct answer.

---

## The Algorithm Step by Step

Here is how Gradient Boosting builds a model from scratch.

**Step 1 — Start with a simple prediction.**

For regression, the initial prediction is just the mean of all target values. For classification, it is the log-odds of the majority class.

$$
F_0(x) = \bar{y} \quad \text{(mean of all target values)}
$$

**Step 2 — Compute the residuals.**

Residuals are the difference between actual values and what the current model predicts. These are the errors the next tree needs to fix.

$$
r_i = y_i - F_{t-1}(x_i)
$$

**Step 3 — Train a shallow tree on the residuals.**

A new decision tree is fitted to predict these residuals, not the original target values. The tree is intentionally kept shallow — typically depth 3 to 5 — to prevent overfitting.

**Step 4 — Add the tree to the ensemble with a learning rate.**

The new tree's predictions are scaled down by a learning rate $\eta$ before being added to the ensemble. This prevents any single tree from making too large a correction.

$$
F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)
$$

Where $h_t(x)$ is the new tree's prediction and $\eta$ is the learning rate.

**Step 5 — Repeat.**

Go back to Step 2. Compute new residuals based on the updated model. Train another tree on them. Add it. Keep going for $T$ rounds.

```
F0: Initial prediction (mean)
  |
  v
Compute residuals r = y - F0
  |
  v
Train tree h1 on residuals
  |
  v
F1 = F0 + η * h1
  |
  v
Compute residuals r = y - F1
  |
  v
Train tree h2 on residuals
  |
  v
F2 = F1 + η * h2
  |
  v
... repeat T times ...
  |
  v
Final model FT = F0 + η*h1 + η*h2 + ... + η*hT
```

---

## The Gradient Part — Why It Is Called Gradient Boosting

The word "gradient" comes from the fact that residuals are actually the **negative gradient of the loss function** with respect to the current predictions.

In regular gradient descent, you update parameters by moving in the direction opposite to the gradient of the loss. In gradient boosting, you update the model itself by adding a tree that points in the direction of the negative gradient.

For MSE loss:

$$
L = \frac{1}{2}(y - F(x))^2
$$

$$
-\frac{\partial L}{\partial F(x)} = y - F(x) = \text{residual}
$$

The residual is the negative gradient. So when you train a tree on residuals, you are essentially doing gradient descent — but in function space rather than parameter space. This generalisation is what makes boosting work with any differentiable loss function, not just MSE.

For a different loss function, you just compute a different gradient. The rest of the algorithm stays exactly the same.

---

## The Learning Rate

The learning rate $\eta$ (also called the shrinkage factor) is one of the most important hyperparameters in gradient boosting.

When each tree's prediction is added to the ensemble, it is multiplied by $\eta$ first. A typical value is between 0.01 and 0.3.

**Large learning rate (e.g. 0.5):** Each tree makes big corrections. The model converges faster but overshoots easily. You need fewer trees but they risk overfitting.

**Small learning rate (e.g. 0.01):** Each tree makes tiny corrections. The model converges slowly but more carefully. You need many more trees but the final model generalises better.

**The key tradeoff:** Learning rate and number of trees are inversely related. A small learning rate needs more trees to reach the same level of performance, but the result is almost always more robust.

```
Large η, few trees:          Small η, many trees:
each step big                each step small
converges fast               converges slowly
less stable                  more stable and accurate
risk of overfitting          better generalisation
```

In practice, it is common to set a small learning rate (0.01 to 0.1) and let early stopping determine the right number of trees.

---

## Shallow Trees — Why Weak Learners Work

Each individual tree in gradient boosting is deliberately kept shallow — depth 3 to 5 is typical. A depth-3 tree can ask at most 3 questions, which means it captures only simple patterns.

This seems counterintuitive. Why use weak models intentionally?

The answer is that you want each tree to correct a little bit of the error, not try to solve the whole problem in one go. If you use deep trees, each one will overfit its own residuals and the ensemble will overfit the training data quickly. Shallow trees correct errors cautiously and collectively — the whole becomes much stronger than any individual part.

This is the opposite philosophy from Random Forest. Random Forest uses full-depth trees (low bias, high variance) and averages them to reduce variance. Gradient Boosting uses shallow trees (high bias, low variance) and adds them to reduce bias.

---

## Bias-Variance in Gradient Boosting

| Stage | Bias | Variance |
|-------|------|----------|
| Too few trees | High | Low | Underfitting |
| Just right | Low | Low | Good generalisation |
| Too many trees | Low | High | Overfitting |

Unlike Random Forest where more trees always help, with gradient boosting you can overfit by adding too many trees. The model starts memorising the training residuals rather than generalising. This is managed using early stopping.

---

## Regularization in Gradient Boosting

Gradient Boosting has several built-in regularization mechanisms.

**Learning rate ($\eta$):** Smaller values prevent any single tree from having too much influence.

**Tree depth (`max_depth`):** Shallow trees cannot overfit as aggressively as deep ones.

**Subsampling:** Like Random Forest's bootstrap sampling, you can train each tree on a random fraction of the training data. This introduces randomness and prevents overfitting. Using less than the full dataset also makes training faster.

**Feature subsampling:** Similar to Random Forest, you can randomly select a subset of features for each tree or even each split. This adds diversity and reduces overfitting.

**Min samples per leaf:** Requiring a minimum number of examples in each leaf prevents the tree from making splits that only satisfy one or two noisy training points.

---

## Popular Gradient Boosting Libraries

The original gradient boosting algorithm is computationally expensive. Several optimised implementations have been developed that are dramatically faster and more powerful.

### XGBoost (Extreme Gradient Boosting)

Introduced in 2016. Dominated machine learning competitions for years. Key improvements over vanilla GBM:

- Uses a more sophisticated tree-building algorithm with second-order gradients (Newton boosting)
- Built-in L1 and L2 regularization on the tree weights
- Parallel processing within each tree (column-based splitting)
- Sparse data handling — ignores missing values automatically
- Built-in cross-validation and early stopping

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=20,
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

### LightGBM (Light Gradient Boosting Machine)

Developed by Microsoft. Significantly faster than XGBoost on large datasets. Key innovations:

- **Histogram-based splitting:** Instead of scanning all possible split values, it bins continuous features into a small number of buckets. Dramatically reduces computation.
- **Leaf-wise tree growth:** Instead of growing trees level by level (depth-wise), LightGBM grows the leaf with the highest loss reduction first. This creates unbalanced but more accurate trees.
- Handles categorical features natively without one-hot encoding

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
)
```

### CatBoost

Developed by Yandex. Designed specifically to handle categorical features without preprocessing. Key innovations:

- **Ordered boosting:** Uses a special technique to prevent target leakage during training — a subtle problem in standard gradient boosting with categorical features
- Handles categorical features natively and extremely well
- Often produces good results out of the box with minimal tuning
- Symmetric trees: each level of the tree uses the same split condition, which speeds up prediction

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    cat_features=['city', 'category', 'brand'],
    early_stopping_rounds=20,
    verbose=0,
    random_state=42
)
model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

### Library Comparison

| | XGBoost | LightGBM | CatBoost |
|--|---------|----------|---------|
| Speed | Fast | Fastest | Fast |
| Memory | Moderate | Low | Moderate |
| Categorical features | Manual encoding needed | Partial support | Native, best-in-class |
| Large datasets | Good | Excellent | Good |
| Tree growth | Depth-wise | Leaf-wise | Symmetric |
| Tuning needed | Moderate | Moderate | Minimal |
| Best for | General purpose | Very large datasets | Data with many categoricals |

---

## Early Stopping

One of the most practical tools in gradient boosting. Instead of specifying the exact number of trees upfront, you let the model keep adding trees and monitor performance on a validation set after each one. When the validation score stops improving for a set number of rounds, training stops automatically.

This prevents overfitting and removes the need to manually tune the number of trees.

```python
model = xgb.XGBClassifier(
    n_estimators=10000,         # set a high maximum
    early_stopping_rounds=50,   # stop if no improvement for 50 rounds
    learning_rate=0.01
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)
# model.best_iteration tells you how many trees were actually used
```

With a small learning rate and early stopping, you get the best of both worlds: careful, stable convergence and automatic selection of the right model size.

---

## Feature Importance in Gradient Boosting

Like Random Forest, gradient boosting provides feature importance scores. Several methods exist:

**Gain:** The average improvement in the loss function brought by each feature across all splits where it was used. This is the most meaningful measure — it tells you how much each feature actually helped reduce error.

**Cover:** The average number of training examples that pass through splits involving each feature. Features that split more data get higher scores.

**Frequency:** How often each feature is used as a split across all trees. Simple count — can be misleading since a frequently used feature is not necessarily the most informative one.

Gain is the most interpretable and the recommended measure.

```python
# XGBoost
importances = model.get_booster().get_score(importance_type='gain')

# General sklearn interface
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
```

---

## Gradient Boosting vs Random Forest

| | Random Forest | Gradient Boosting |
|--|--------------|------------------|
| Tree building | Parallel (independent) | Sequential (each learns from previous) |
| Tree depth | Full depth (deep) | Shallow (depth 3 to 5) |
| Bias | Low | Starts high, reduces with each round |
| Variance | Low (averaging reduces it) | Can increase with too many trees |
| Overfitting risk | Low | Moderate — needs careful tuning |
| Sensitivity to outliers | Moderate | Higher |
| Training speed | Faster (parallel) | Slower (sequential) |
| Hyperparameter tuning | Simpler | More involved |
| Performance on tabular data | Strong | Often stronger |
| Interpretability | Low | Low |

In practice on structured tabular data, gradient boosting (especially XGBoost or LightGBM) tends to outperform Random Forest when properly tuned. Random Forest is faster to get working and more forgiving of poor hyperparameter choices. Gradient boosting rewards careful tuning with better results.

---

## When to Use Gradient Boosting

Gradient Boosting is the right choice when:

- You are working with structured tabular data (spreadsheet-style)
- Predictive accuracy is the top priority
- You have enough time to tune hyperparameters carefully
- You are in a data science competition — GBM variants dominate leaderboards
- The dataset is medium to large (thousands to millions of examples)
- You need feature importance for understanding the data

It is not the best choice when:

- You need high interpretability — single decision trees or logistic regression are better
- The dataset is very small — gradient boosting can overfit on tiny datasets
- Training speed is critical — Random Forest trains faster
- You are working with image, audio, or text data — neural networks are better suited
- You need fast iteration with minimal tuning — Random Forest is more forgiving

---

## Implementation in Code (sklearn)

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    max_features='sqrt',
    min_samples_leaf=10,
    random_state=42
)
model.fit(X_train, y_train)

print(f"Train accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {model.score(X_test, y_test):.3f}")

# Feature importance
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 features:")
print(importances.head(10))
```

---

## The Full Picture

```
Training data (features X, targets y)
              |
              v
F0 = mean(y)   <-- initial prediction
              |
    __________|__________
   |                     |
   For t = 1 to T trees:
   |
   v
   Compute residuals:
   r = y - F(t-1)(x)          <-- what the current model gets wrong
   |
   v
   Fit shallow tree h_t on residuals
   (tree learns to predict the errors)
   |
   v
   Update model:
   F_t = F_(t-1) + η * h_t    <-- small correction added
   |
   v
   Check validation score
   (stop early if no improvement)
   |
   v
   Repeat with new residuals
              |
              v
Final model = F0 + η*h1 + η*h2 + ... + η*hT
(sum of many small corrections)
              |
              v
New input → pass through all T trees
           sum their predictions
           → final output
```

---

## Quick Concept Summary

| Concept | What It Means |
|---------|--------------|
| Gradient Boosting | Building trees sequentially where each tree corrects the errors of the previous ensemble |
| Residuals | The difference between actual values and current predictions — what the next tree tries to predict |
| Weak learner | A shallow decision tree that is only slightly better than random — the building block of GBM |
| Learning rate ($\eta$) | Scales each tree's contribution — smaller means more careful, more trees needed |
| Sequential ensemble | Trees built one after another, each depending on the previous |
| Gradient descent in function space | What boosting is mathematically — moving in the direction of the negative gradient of the loss |
| Early stopping | Automatically halting training when validation performance stops improving |
| Subsampling | Training each tree on a random fraction of data to add diversity and prevent overfitting |
| XGBoost | Optimised GBM with regularization and parallel tree building |
| LightGBM | Fast GBM using histogram-based splits and leaf-wise tree growth |
| CatBoost | GBM with native categorical feature handling and ordered boosting |
| Feature importance (Gain) | Average improvement in loss when a feature is used in a split |
