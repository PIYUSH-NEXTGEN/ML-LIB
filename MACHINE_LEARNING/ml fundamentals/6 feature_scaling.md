# Feature Scaling in Machine Learning

Feature scaling is the process of **bringing all features onto a similar numerical scale** before feeding them into a machine learning model.

Most real-world datasets have features measured in completely different units and ranges:

| Feature | Range |
|---------|-------|
| House size (sq ft) | 500 – 5,000 |
| Number of rooms | 1 – 10 |
| Age of house (years) | 0 – 100 |
| Price (₹) | 10,00,000 – 2,00,00,000 |

These features live on wildly different scales. Without scaling, this causes serious problems during training.

---

## Why Does Scale Mismatch Cause Problems?

### Problem 1 — Gradient Descent Becomes Inefficient

Gradient Descent updates weights based on gradients. When features have very different scales, the cost function becomes uneven.

The model wastes steps bouncing back and forth instead of moving cleanly toward the minimum. Scaling makes the bowl round, so Gradient Descent converges **faster and more smoothly**.

### Problem 2 — Large Features Dominate

A feature with values in the thousands naturally produces larger gradients than a feature with values between 0 and 1. The model ends up paying more attention to large-valued features  not because they're more important, but simply because their numbers are bigger.

**Example:** House size (500–5000) will overshadow number of rooms (1–10), even if rooms matter more for the price.

---

## Methods of Feature Scaling

### 1. Min-Max Normalisation (Normalization)

Scales all values to a fixed range, typically **0 to 1**.

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

| Original Value | Min | Max | Scaled Value |
|---------------|-----|-----|-------------|
| 1500 | 500 | 5000 | $\frac{1500 - 500}{5000 - 500} = 0.22$ |
| 3000 | 500 | 5000 | $\frac{3000 - 500}{5000 - 500} = 0.56$ |
| 5000 | 500 | 5000 | $\frac{5000 - 500}{5000 - 500} = 1.00$ |

**Result:** Every feature is now between 0 and 1.

**Use when:** You know the data has fixed, meaningful bounds (pixel values 0–255, age, percentage scores).

**Weakness:** Sensitive to outliers. One extreme value compresses all other values into a tiny range.

---

### 2. Standardisation (Z-Score Normalisation)

Scales values so the feature has **mean = 0** and **standard deviation = 1**.

$$
x' = \frac{x - \mu}{\sigma}
$$

Where:
- $\mu$ = mean of the feature
- $\sigma$ = standard deviation of the feature

**Example:** House sizes with mean = 2000, std = 800:

| Original | Scaled |
|----------|--------|
| 1200 | $\frac{1200 - 2000}{800} = -1.0$ |
| 2000 | $\frac{2000 - 2000}{800} = 0.0$ |
| 3600 | $\frac{3600 - 2000}{800} = 2.0$ |

**Result:** Values are centred around 0, most falling between -3 and +3.

**Use when:** Data has outliers, or you don't know the natural bounds. Works well for most ML algorithms.

**Strength:** Outliers don't destroy the scaling they just end up with high z-scores.

---

### 3. Mean Normalisation

A variation of min-max that centres values around 0.

$$
x' = \frac{x - \mu}{x_{max} - x_{min}}
$$

**Result:** Values fall roughly between -0.5 and 0.5, centred at 0.

Less common than the two above, but sometimes used in practice when you want both centring and bounded range.

---

## Min-Max vs Standardisation — Which to Use?

| | Min-Max Normalisation | Standardisation |
|--|----------------------|----------------|
| **Output range** | Fixed: 0 to 1 | Unbounded, centred at 0 |
| **Sensitive to outliers** | Yes | No |
| **When to use** | Known bounds, neural networks, image pixel values | Most other cases, data with outliers |
| **Preserves zero values** | No | No |

**General rule:** When in doubt, start with **Standardisation**. It works well across most algorithms and is more robust to outliers.

---

## Which Algorithms Need Feature Scaling?

Not every algorithm is affected by feature scale.

### Needs Scaling ✅

| Algorithm | Why |
|-----------|-----|
| Linear Regression | Gradient Descent is scale-sensitive |
| Logistic Regression | Same reason |
| Neural Networks | Unstable training without scaling |
| K-Nearest Neighbours (KNN) | Uses distance — large features dominate |
| K-Means Clustering | Uses distance — same issue |
| Support Vector Machines (SVM) | Margin calculation is scale-sensitive |
| Principal Component Analysis (PCA) | Variance is scale-dependent |

### Does Not Need Scaling ❌

| Algorithm | Why |
|-----------|-----|
| Decision Trees | Split on thresholds, not magnitude |
| Random Forest | Ensemble of trees — same reason |
| Gradient Boosting (XGBoost) | Tree-based, scale invariant |
| Naive Bayes | Works with probabilities, not distances |

---

```
✅ Correct:
   Compute mean and std from training data
   Scale training data using those values
   Scale test data using the SAME values

❌ Wrong:
   Compute mean and std separately for test data
   (This leaks test information into your pipeline — data leakage)
```

### Why?

In the real world, when a new data point arrives, you won't know its min/max. You only have the statistics from training. Fitting the scaler on test data gives an unrealistically clean result — your model appears to perform better than it actually will in production.

---

## Quick Concept Summary

| Concept | What It Is | Simple Analogy |
|---------|-----------|----------------|
| **Feature Scaling** | Bringing features to a similar range | Converting km and miles to the same unit |
| **Min-Max Normalisation** | Compress values to 0–1 | Rescaling a 0–100 score to 0–1 |
| **Standardisation** | Centre at 0 with std = 1 | How many standard deviations from the average? |
| **Data Leakage** | Using test data info during training | Peeking at the exam before sitting it |
| **Scale Sensitivity** | When large values dominate gradients | A louder voice drowning out a quieter one |
