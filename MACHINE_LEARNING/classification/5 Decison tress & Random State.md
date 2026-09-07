# Decision Trees and Random Forest 

A decision tree is a machine learning model that makes predictions by asking a series of questions about the input features, one question at a time, until it arrives at an answer. The structure looks exactly like a flowchart — you start at the top, follow the path based on your answers, and land on a prediction at the bottom.

It is one of the most intuitive machine learning algorithms because the entire decision-making process is visible and readable. You do not need a math degree to understand why a decision tree made a particular prediction. You just read the path it took.

Decision trees work for both classification (predicting a category) and regression (predicting a number).

---

## A Concrete Example

Suppose you want to predict whether a person will buy a product based on their age, income, and whether they are a returning customer.

```
Is the person a returning customer?
        |
      Yes |                   No
        |                      |
Is income > 50K?         Is age < 30?
   |         |               |         |
  Yes        No             Yes        No
   |          |              |          |
 Buys      Buys          Buys       Does Not
           (50%)                      Buy
```

Each internal node asks a question about one feature. Each branch represents the answer. Each leaf node at the bottom gives a prediction. This is a decision tree.

---

## Key Terminology

**Root Node:** The very first question at the top of the tree. It splits the entire dataset.

**Internal Node (Decision Node):** Any node that asks a question and splits the data further.

**Leaf Node (Terminal Node):** A node at the bottom that gives a final prediction. No more splitting happens here.

**Branch:** The path taken after answering a question at a node.

**Depth:** How many levels of questions the tree has. A tree with depth 3 asks at most 3 questions before giving an answer.

**Splitting:** The process of dividing the data at a node into two or more subgroups based on a feature.

**Pruning:** Removing branches from a trained tree to reduce overfitting and simplify the model.

---

## How Does a Decision Tree Decide Where to Split?

This is the core of how a decision tree learns. At each node, the algorithm considers every feature and every possible threshold for that feature, then picks the one that creates the **purest** subgroups after splitting.

Purity means that after the split, the data in each branch is as uniform as possible — ideally all one class. If you split a node and one branch ends up with 100% class A and the other ends up with 100% class B, that is a perfect split. If both branches still have a 50/50 mix of classes, that split was useless.

There are two main measures of impurity used to evaluate splits.

### Gini Impurity

Gini impurity measures the probability of incorrectly classifying a randomly chosen element if it were labelled according to the class distribution in the node.

$$
Gini = 1 - \sum_{k=1}^{K} p_k^2
$$

Where $p_k$ is the proportion of class $k$ in the node.

**Gini = 0:** The node is completely pure — all examples belong to one class. Perfect.

**Gini = 0.5:** Maximum impurity for binary classification — classes are perfectly mixed 50/50.

**Example:**

A node has 40 examples: 30 class A and 10 class B.

$$
p_A = \frac{30}{40} = 0.75, \quad p_B = \frac{10}{40} = 0.25
$$

$$
Gini = 1 - (0.75^2 + 0.25^2) = 1 - (0.5625 + 0.0625) = 1 - 0.625 = 0.375
$$

After a split, the algorithm checks whether the weighted average Gini of the two child nodes is lower than the parent's Gini. If it is, the split improved purity.

### Entropy and Information Gain

Entropy comes from information theory. It measures the disorder or uncertainty in a node.

$$
Entropy = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

**Entropy = 0:** Perfect purity. No uncertainty.

**Entropy = 1:** Maximum disorder for binary classification. Complete 50/50 mix.

**Information Gain** is the reduction in entropy achieved by a split:

$$
IG = Entropy_{parent} - \sum_{child} \frac{n_{child}}{n_{parent}} \cdot Entropy_{child}
$$

The algorithm picks the split with the highest information gain — the one that reduces uncertainty the most.

**Example with the same node (30 class A, 10 class B):**

$$
Entropy = -(0.75 \log_2 0.75 + 0.25 \log_2 0.25)
$$
$$
= -(0.75 \times (-0.415) + 0.25 \times (-2))
= -(- 0.311 - 0.5) = 0.811
$$

### Gini vs Entropy

In practice, they usually give very similar results. Gini is slightly faster to compute (no logarithm) and is the default in sklearn. Entropy sometimes gives slightly more balanced trees. The choice rarely matters much.

### For Regression Trees — Variance Reduction

For regression tasks, instead of measuring class impurity, the tree measures how spread out the output values are in each node. It picks the split that most reduces the variance of values in the child nodes.

$$
\text{Variance} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2
$$

The leaf node prediction for a regression tree is simply the **mean** of all training examples that ended up in that leaf.

---

## How the Tree Grows

Starting from the root, the decision tree grows greedily:

```
Start with all training data at the root node
              |
              v
Try every feature and every threshold
              |
              v
Pick the split that gives lowest Gini
(or highest Information Gain)
              |
              v
Split the data into two child nodes
              |
              v
Repeat the process for each child node
              |
              v
Stop when a stopping condition is met:
  - Node is perfectly pure (Gini = 0)
  - Node has fewer than min_samples_split examples
  - Maximum depth has been reached
  - Splitting no longer improves purity meaningfully
```

This is called a **greedy** algorithm because at each node it picks the locally best split without looking ahead at what future splits might look like. It does not guarantee the globally optimal tree, but it works well in practice.

---

## Overfitting in Decision Trees

A decision tree that is allowed to grow without any restrictions will keep splitting until every leaf node is completely pure — often containing just one training example. This is severe overfitting.

The tree memorises the training data perfectly but fails completely on new data because it has learned the noise rather than the signal.

```
Depth 2 (underfitting):     Depth 20 (overfitting):

    simple splits               every training point
    smooth boundary             in its own leaf
    misses some patterns        fits training perfectly
    high train error            fails on new data
    high test error             very low train error
                                very high test error
```

### Controlling Tree Complexity

| Hyperparameter | What It Does |
|----------------|-------------|
| `max_depth` | Maximum number of levels allowed in the tree |
| `min_samples_split` | Minimum examples required to split a node |
| `min_samples_leaf` | Minimum examples required in a leaf node |
| `max_features` | Maximum number of features considered at each split |
| `min_impurity_decrease` | Minimum purity improvement required to make a split |

These hyperparameters are the primary tools for preventing overfitting. They are tuned using cross-validation.

### Pruning

Post-pruning is an alternative approach. You grow the full tree first, then work backwards and remove branches that do not significantly improve performance on a validation set. The idea is to collapse leaf nodes that add complexity without adding much predictive power.

---

## Strengths and Weaknesses of Decision Trees

**Strengths:**

- Completely interpretable — you can read and explain every decision
- Handles both numerical and categorical features naturally
- Requires no feature scaling
- Handles non-linear relationships automatically
- Fast to train and predict

**Weaknesses:**

- Highly prone to overfitting without careful tuning
- High variance — small changes in training data can produce completely different trees
- Tends to create biased trees when classes are imbalanced
- Captures only axis-aligned splits — diagonal boundaries require many splits

The high variance problem is the most important weakness. It leads directly to the motivation for Random Forest.

---

## Part 2: Random Forest

A Random Forest is an ensemble of many decision trees, each trained slightly differently, whose predictions are combined to produce a final answer. It is one of the most powerful and widely used machine learning algorithms.

The name comes from the idea of building many trees and having them vote together. A single tree is unstable and prone to overfitting. A forest of many diverse trees is stable and robust. The diversity is what matters — if all trees were identical, combining them would change nothing.

Random Forest was introduced by Leo Breiman in 2001 and remains competitive with much more complex algorithms on structured, tabular data.

---

## The Problem Random Forest Solves

Decision trees have high variance. Train the same tree algorithm on two slightly different subsets of the same dataset and you can get completely different trees. This instability means a single tree is unreliable.

The solution is **ensemble learning** — combine many models. When many unstable models vote together, their individual errors cancel out and the collective prediction is much more stable and accurate.

But there is a catch. If you train many trees on the same data using the same algorithm, they will all look similar and make similar errors. Their votes will be correlated. Combining correlated models does not help much.

Random Forest breaks this correlation using two sources of randomness.

---

## Two Sources of Randomness

### 1. Bootstrap Sampling (Bagging)

Each tree in the forest is trained on a different random sample of the training data. The sample is drawn **with replacement** — this is called bootstrapping. With replacement means the same example can be picked multiple times, and some examples will not be picked at all.

With $m$ training examples and sampling with replacement, roughly 63% of unique examples end up in each bootstrap sample. The remaining 37% are called **out-of-bag (OOB)** examples — these can be used to evaluate each tree without needing a separate validation set.

```
Original dataset: [A, B, C, D, E, F, G, H]

Bootstrap sample for tree 1: [A, C, C, E, F, A, H, B]  (with replacement)
Bootstrap sample for tree 2: [B, D, G, G, A, C, F, E]
Bootstrap sample for tree 3: [C, H, B, D, D, F, A, G]

Each tree sees a different version of the data.
```

### 2. Random Feature Selection

At each split in each tree, instead of considering all $n$ features, the algorithm considers only a **random subset** of features. The split is chosen from this subset only.

For classification the typical subset size is $\sqrt{n}$ features. For regression it is $\frac{n}{3}$ features.

This is the key innovation that makes trees truly diverse. Even when two trees are trained on similar bootstrap samples, the random feature selection forces them to use different features at different splits. They end up structurally different and make different types of errors.

```
Feature set: [size, rooms, age, location, floor, distance]  (6 features)

Tree 1, node 1: randomly selects [size, age, floor]       → best split on size
Tree 1, node 2: randomly selects [rooms, location, floor]  → best split on location

Tree 2, node 1: randomly selects [rooms, age, distance]   → best split on age
Tree 2, node 2: randomly selects [size, rooms, distance]   → best split on size

Each tree builds a different structure even from similar data.
```

---

## Making Predictions with a Random Forest

### Classification

Every tree votes for a class. The class with the most votes wins.

```
100 trees predicting whether email is spam:

Tree 1:  spam
Tree 2:  not spam
Tree 3:  spam
Tree 4:  spam
...
Tree 100: spam

68 votes for spam, 32 votes for not spam
Final prediction: spam
```

### Regression

Every tree predicts a numerical value. The final prediction is the average of all trees.

```
5 trees predicting house price:

Tree 1:  ₹45L
Tree 2:  ₹52L
Tree 3:  ₹47L
Tree 4:  ₹49L
Tree 5:  ₹51L

Final prediction: (45 + 52 + 47 + 49 + 51) / 5 = ₹48.8L
```

---

## Out-of-Bag Error Estimation

Since each tree is trained on only about 63% of the data, the remaining 37% (out-of-bag examples) can be used to estimate the model's performance without a separate validation set.

For each training example, collect predictions only from trees that did not see that example during training. Average these predictions (or take a majority vote) and compare to the true label. This gives the **out-of-bag error**, which is a reliable estimate of generalisation performance.

This is one of the practical conveniences of Random Forest — you get a free performance estimate without burning a validation set.

---

## Feature Importance

One of the most useful outputs of a Random Forest is **feature importance scores**. These tell you which features the forest relied on most heavily across all its trees.

The default method (Mean Decrease in Impurity) works by accumulating how much each feature reduced Gini impurity across all splits across all trees. Features used early in many trees and that produce clean splits get high importance scores.

```python
importances = pd.Series(
    forest.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
```

This is a practical and widely used tool for understanding which inputs matter most and for guiding feature selection.

---

## Key Hyperparameters of Random Forest

| Hyperparameter | What It Controls | Typical Starting Point |
|----------------|-----------------|----------------------|
| `n_estimators` | Number of trees in the forest | 100 to 500 |
| `max_depth` | Maximum depth of each tree | None (grow fully) or 10 to 20 |
| `max_features` | Features considered at each split | sqrt(n) for classification |
| `min_samples_split` | Minimum examples to split a node | 2 |
| `min_samples_leaf` | Minimum examples in a leaf | 1 |
| `bootstrap` | Whether to use bootstrap sampling | True |
| `oob_score` | Whether to compute out-of-bag error | True (free validation) |

More trees generally means better and more stable performance, up to a point where adding more trees gives diminishing returns. Unlike a single decision tree, Random Forest is relatively robust to hyperparameter choices — a reasonable default configuration often works well.

---

## Why Random Forest Does Not Overfit as Badly

A single decision tree overfits because it memorises the training data with high-variance splits. Adding more trees to a Random Forest actually stabilises the model rather than making it overfit more.

As you add more trees, the predictions average out. Individual trees may overfit their specific bootstrap sample, but their errors are random and uncorrelated. When you average many uncorrelated noisy predictions, the noise cancels out and the signal remains.

Mathematically, if each tree has variance $\sigma^2$ and the trees are uncorrelated, the variance of the average of $T$ trees is $\frac{\sigma^2}{T}$. More trees means lower variance — the opposite of what happens with a single deeper tree.

The random feature selection is what ensures the trees remain sufficiently uncorrelated. Without it, the trees would all look similar and averaging them would not help much.

---

## Decision Tree vs Random Forest

| | Decision Tree | Random Forest |
|--|--------------|--------------|
| Number of models | 1 | Many (typically 100 to 500) |
| Training data | Full dataset | Bootstrap sample per tree |
| Feature selection at split | All features | Random subset |
| Interpretability | Very high | Low (many trees) |
| Variance | High | Low |
| Overfitting | Prone | Resistant |
| Training speed | Fast | Slower (trains many trees) |
| Prediction speed | Very fast | Slower (aggregates many trees) |
| Performance | Moderate | High |
| Feature importance | Yes | Yes (more reliable) |
| Requires scaling | No | No |

---

## When to Use Each

**Use a Decision Tree when:**

- Interpretability is the highest priority — you need to explain every decision to a stakeholder
- The dataset is small and you cannot afford ensemble overhead
- You are building a quick baseline model
- Rules extracted from the model will be used directly in a system

**Use Random Forest when:**

- You want strong predictive performance on tabular data
- Interpretability is less important than accuracy
- You want feature importance scores to understand your data
- You are dealing with moderate to high-dimensional data
- Overfitting is a concern with a single tree

---

## Implementation in Code

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree — Classification
dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    criterion='gini',
    random_state=42
)
dt.fit(X_train, y_train)
print(f"Train accuracy: {dt.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {dt.score(X_test, y_test):.3f}")

# Random Forest — Classification
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    max_features='sqrt',
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print(f"OOB score:     {rf.oob_score_:.3f}")
print(f"Test accuracy: {rf.score(X_test, y_test):.3f}")

# Feature importance from Random Forest
importances = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print("\nTop features:")
print(importances.head(10))
```

---

## The Full Picture

```
Decision Tree:

Training data
      |
      v
Greedily find best split (Gini or Entropy)
at each node, one at a time
      |
      v
Grow tree until stopping condition met
      |
      v
Single tree with rules you can read
      |
      v
High variance — changes with data


Random Forest:

Training data
      |
      v
For each of T trees:
    Bootstrap sample the data
         |
         v
    At each split:
    Randomly select sqrt(n) features
         |
         v
    Grow full decision tree on this sample
      |
      v
Aggregate T trees:
  Classification → majority vote
  Regression     → average
      |
      v
Stable, accurate model
Low variance, resistant to overfitting
```

---

## Quick Concept Summary

| Concept | What It Means |
|---------|--------------|
| Decision Tree | A model that makes predictions by following a series of if-then questions |
| Gini Impurity | Measures how mixed the classes are at a node — lower is purer |
| Entropy | Measures uncertainty at a node — lower means more certain |
| Information Gain | How much a split reduces entropy — higher is better |
| Leaf Node | The final node giving a prediction — no more splits |
| Pruning | Removing branches to simplify the tree and reduce overfitting |
| Random Forest | An ensemble of many diverse decision trees voting together |
| Bootstrap Sampling | Training each tree on a random sample drawn with replacement |
| Random Feature Selection | Considering only a random subset of features at each split |
| Out-of-Bag Error | Free validation estimate using the examples each tree did not see |
| Feature Importance | Score showing how much each feature contributed across all trees |
| Ensemble | Combining many models to produce a stronger, more stable prediction |
| Bagging | Bootstrap Aggregating — the technique Random Forest uses to create diversity |