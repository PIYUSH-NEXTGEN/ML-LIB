# Support Vector Machine (SVM)
A Support Vector Machine is a supervised machine learning algorithm that finds the best possible boundary between classes in your data. It works for both classification and regression, but it is most naturally understood through classification.

The central idea is simple but powerful. When you have two classes of data points, there are usually many possible lines (or curves) that can separate them. SVM does not just find any boundary — it finds the one that is **as far away from both classes as possible**. This distance from the boundary to the nearest data points on each side is called the **margin**, and SVM maximises it.

This focus on the margin is what makes SVM particularly good at generalising. A boundary that sits comfortably away from all training points tends to handle new, unseen points better than one squeezed close to the data.

---

## The Intuition Behind Maximum Margin

Imagine two groups of points on a piece of paper and you need to draw a line separating them. You could draw thousands of valid lines. But which one would you trust most for future data?

The safest line is the one that keeps the most breathing room between itself and both groups. If a new point arrives slightly different from the training examples, a line with large breathing room is more likely to still classify it correctly.

```
Poor boundary (too close to one class):

  o o                         
  o o   |  x x                  
  o o   |  x x                  
  o o   |  x x                  

Best boundary (maximum margin):

  o o                       
  o o  - - - | - - -  x x   
  o o        |        x x   
             |              
         max margin
```

SVM finds that maximally-spaced boundary automatically.

---

## Key Terminology

**Hyperplane:** The decision boundary in SVM. In 2D it is a line. In 3D it is a plane. In higher dimensions it is called a hyperplane. This is what separates the two classes.

**Margin:** The total distance between the two parallel boundary lines on either side of the hyperplane. SVM maximises this.

**Support Vectors:** The specific training data points that lie closest to the hyperplane, right on the edge of the margin. These are the only points that actually determine where the hyperplane sits. If you removed every other training point and kept only the support vectors, the model would be identical.

**Maximum Margin Classifier:** What SVM is trying to build — a hyperplane with the widest possible margin.

```
         o                       x
      o     o                 x     x
            o   |  margin  |  x
            o <-- margin-->x
            o   |          |  x
      o     o                 x     x
         o     support          x
               vectors
               (closest points to the boundary on each side)
```

The support vectors are the critical points. Everything else is irrelevant once the boundary is found.

---

## Hard Margin vs Soft Margin

### Hard Margin SVM

The original formulation of SVM requires that all training points are correctly classified with no exceptions — every point must sit outside the margin on the correct side. This is called a **hard margin**.

Hard margin works only when the data is **linearly separable** — meaning a straight line (or hyperplane) can perfectly separate the two classes with zero errors.

```
Linearly separable:       Not linearly separable:
  o o | x x               o x o x o
  o o | x x               x o x o x
  o o | x x               o x o x o

Hard margin works here.   Hard margin fails here.
```

In reality, data is almost never perfectly separable. There is always some noise, overlap, or mislabelled points. A hard margin SVM would either fail to find a boundary or fit wildly to accommodate the noisy points.

### Soft Margin SVM

Soft margin SVM introduces flexibility. It allows some training points to violate the margin — to sit inside it or even on the wrong side — in exchange for finding a better overall boundary.

This flexibility is controlled by a hyperparameter called **C**.

**Slack variables ($\xi$)** are introduced for each training point. They measure how much that point violates the margin. A point correctly classified and outside the margin has $\xi = 0$. A point inside the margin has $\xi > 0$. A point on the wrong side has $\xi > 1$.

The soft margin cost function balances two things:

$$
\text{Minimise: } \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{m} \xi_i
$$

| Term | What It Does |
|------|-------------|
| $\frac{1}{2} \|w\|^2$ | Maximise the margin (smaller $w$ means wider margin) |
| $C \sum \xi_i$ | Penalise margin violations |
| $C$ | Controls the tradeoff between margin width and violations |

### The Role of C

C is the most important hyperparameter in SVM. It controls how strict the model is about misclassifications during training.

**Large C:** The model is heavily penalised for any violation. It tries hard to classify every training point correctly. This leads to a narrower margin and potential overfitting — the boundary hugs the training data closely.

**Small C:** Violations are tolerated more easily. The model prioritises a wider margin even if it means some training points are misclassified. This leads to better generalisation — the boundary is smoother and less influenced by individual points.

```
Large C (low tolerance):      Small C (high tolerance):

  o o |x x                      o o   |   x x
  o o|x x                       o o   |   x x
  o o|x x                       o o   |   x x

Narrow margin,                Wide margin,
fits training closely,        more robust,
risk of overfitting           risk of underfitting
```

Finding the right C requires cross-validation.

---

## The Kernel Trick

So far we have only discussed linear boundaries. But most real-world data is not linearly separable — the classes are mixed in complicated ways that no straight line can separate.

The **kernel trick** is what makes SVM powerful enough to handle non-linear data. It transforms the data into a higher-dimensional space where a linear boundary can separate the classes, then maps the boundary back to the original space where it appears as a curve.

The key insight is that SVM never actually needs to compute the coordinates of points in that higher-dimensional space. It only needs the **dot products** between pairs of points. A kernel function computes these dot products directly without the expensive transformation.

```
Original 2D space              Transformed 3D space
(not linearly separable):      (linearly separable):

  o x o x o                        x x x
  x o x o x          →         o o      o o
  o x o x o                        x x x

A line cannot separate them.   A plane can separate them.
Map the boundary back           → curved boundary in 2D.
```

### Common Kernels

#### Linear Kernel

$$
K(x, z) = x^T z
$$

Just the standard dot product. No transformation. This is identical to a standard linear SVM. Use it when the data is already linearly separable or has many features (high-dimensional data like text is often linearly separable).

#### Polynomial Kernel

$$
K(x, z) = (x^T z + c)^d
$$

Implicitly maps the data to a polynomial feature space of degree $d$. Equivalent to adding polynomial features like you would in polynomial regression, but done efficiently without creating them explicitly.

| Parameter | Meaning |
|-----------|---------|
| $d$ | Degree of the polynomial |
| $c$ | Constant controlling influence of higher-degree terms |

#### RBF Kernel (Radial Basis Function) — The Default

$$
K(x, z) = \exp\left(-\gamma \|x - z\|^2\right)
$$

Also called the Gaussian kernel. This is the most widely used kernel in practice. It measures similarity between two points based on distance — nearby points get a high score, far-apart points get a score near zero.

The RBF kernel maps data into an infinite-dimensional space, which sounds extreme but works remarkably well. It can model very complex, non-linear boundaries.

**Gamma ($\gamma$)** controls how far the influence of a single training point reaches:

**Large $\gamma$:** Each point has very local influence. The boundary tightly wraps around individual training points. Risk of overfitting.

**Small $\gamma$:** Each point has broad influence. The boundary is smooth and generalised. Risk of underfitting.

```
Large gamma:           Small gamma:
  boundary wraps         boundary is smooth
  tightly around         and generalised
  each point

  o oo|xx x             o o  |  x x
  o o |x x x            o o  |  x x
  o oo|xx                o o  |  x x
```

#### Sigmoid Kernel

$$
K(x, z) = \tanh(\gamma x^T z + c)
$$

Similar to the activation function in neural networks. Less commonly used than RBF or polynomial but occasionally works well for certain types of data.

### Kernel Comparison

| Kernel | Use When |
|--------|----------|
| Linear | Data is linearly separable or has many features |
| Polynomial | Known polynomial relationship in data |
| RBF | Default choice for non-linear data, unknown relationship |
| Sigmoid | Sometimes for neural-network-like behaviour |

---

## SVM for Regression — SVR

SVM can also be used for regression tasks. This is called **Support Vector Regression (SVR)**.

In classification, SVM tries to keep data points outside the margin. In regression, SVR flips this: it tries to keep predictions **inside** a tube of width $\epsilon$ around the true values. Points that fall within the tube contribute zero error. Only points that fall outside the tube are penalised.

```
SVR tube:

  Actual values with tube:

         |   ............   |
  y      |  . true curve .  |
         | .              . |
  -------|----------------|-------  tube boundary
         | .              . |
         |  . true curve .  |
         |   ............   |

Points inside the tube: zero error
Points outside the tube: penalised
```

This makes SVR robust to small noise in the data — minor deviations from the true values are simply ignored.

---

## SVM for Multi-Class Classification

SVM is inherently a binary classifier (two classes). For problems with more than two classes, two strategies are used:

**One vs Rest (OvR):** Train one SVM per class. Each SVM learns to separate its class from all others. To predict, run all SVMs and pick the one that classifies the new point with the highest confidence.

```
3 classes: cat, dog, bird

SVM 1: cat vs (dog + bird)
SVM 2: dog vs (cat + bird)
SVM 3: bird vs (cat + dog)

New point → run through all 3 → pick winner
```

**One vs One (OvO):** Train one SVM for every pair of classes. For K classes you get $\frac{K(K-1)}{2}$ SVMs. To predict, run all of them and pick the class that wins the most pairwise matchups.

```
3 classes: cat, dog, bird

SVM 1: cat vs dog
SVM 2: cat vs bird
SVM 3: dog vs bird

New point → each SVM votes → class with most votes wins
```

One vs One is more commonly used in practice because each individual SVM trains on a smaller, cleaner subset of the data.

---

## Feature Scaling Is Essential for SVM

SVM computes distances between points to find the support vectors and maximise the margin. If features are on very different scales, the ones with larger values dominate the distance calculation completely.

A feature ranging from 0 to 10,000 will overwhelm a feature ranging from 0 to 1, making the second feature effectively invisible to the model.

Always standardise features before training an SVM.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

---

## When to Use SVM

SVM works well when:

- The dataset is small to medium sized (a few hundred to tens of thousands of examples)
- The number of features is large relative to the number of examples (text classification, genomics)
- You need a clear margin of separation and the data is roughly linearly separable
- You want a model that is robust to outliers (support vectors determine the boundary, not all points)
- The relationship between features and output is non-linear (use RBF kernel)

SVM is not a great choice when:

- The dataset is very large — training SVM scales poorly with data size ($O(m^2)$ to $O(m^3)$ complexity)
- You need fast training and prediction — neural networks or tree-based models are faster on large data
- Features and target have a very noisy relationship — the margin concept becomes hard to apply
- You need probability estimates — SVM does not naturally output probabilities (Platt scaling can add this but it is slow)

---

## Implementation in Code

```python
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Always scale before SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Classification with RBF kernel (default)
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X_train_scaled, y_train)
predictions = clf.predict(X_test_scaled)

# Check how many support vectors were selected
print(f"Number of support vectors: {clf.n_support_}")

# Regression
reg = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
reg.fit(X_train_scaled, y_train)

# Tuning C and gamma with cross-validation
param_grid = {
    'C':     [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'linear']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Best C:      {grid_search.best_params_['C']}")
print(f"Best gamma:  {grid_search.best_params_['gamma']}")
print(f"Best kernel: {grid_search.best_params_['kernel']}")
```

---

## The Full Picture

```
Training data (two classes)
          |
          v
Find the hyperplane that maximises margin
          |
     _____|_____
    |           |
Data is      Data is not
linear       linear
    |           |
    v           v
Linear       Apply kernel trick
SVM          (RBF, Polynomial, etc.)
             Transform to higher
             dimensional space where
             data is linearly separable
                  |
                  v
          Find linear hyperplane
          in transformed space
                  |
                  v
          Map back to original space
          = curved decision boundary
                  |
                  v
          Support Vectors identified
          (closest points to boundary)
                  |
                  v
          Tune C (and gamma for RBF)
          via cross-validation
                  |
                  v
          Trained SVM model
          ready for inference
```

---

## Quick Concept Summary

| Concept | What It Means |
|---------|--------------|
| Hyperplane | The decision boundary that separates the two classes |
| Margin | The gap between the boundary and the closest points on each side |
| Support Vectors | The training points sitting right on the edge of the margin — the only ones that define the boundary |
| Hard Margin | No violations allowed — only works on perfectly separable data |
| Soft Margin | Some violations allowed — works on real messy data |
| C | Controls tolerance for violations — large C means narrow margin, small C means wide margin |
| Kernel Trick | Implicitly maps data to higher dimensions so a linear boundary becomes a curve in the original space |
| RBF Kernel | Default kernel — measures similarity by distance, works well on most non-linear problems |
| Gamma | Controls influence range of each training point in RBF — large gamma overfits, small gamma underfits |
| SVR | SVM adapted for regression — fits a tube around the data instead of a boundary between classes |
| Feature Scaling | Mandatory before SVM — distance-based algorithm is sensitive to feature magnitude |
