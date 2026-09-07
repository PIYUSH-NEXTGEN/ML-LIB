# Preprocessing Techniques

Raw data is almost never ready to feed into a machine learning model. It is messy, inconsistent, and often structured in ways that confuse algorithms. Preprocessing is the work you do to clean, transform, and prepare data before training begins. The quality of this step directly determines the quality of the model.

---

## 1. Data Cleaning

Data cleaning is the process of fixing or removing incorrect, corrupted, duplicate, or unusable records from your dataset. Garbage in, garbage out  no model can compensate for fundamentally broken data.

### Missing Values

Missing values are one of the most common issues in real datasets. A sensor skipped a reading. A survey respondent left a question blank. A database field was never filled in.

You have three main options:

**Drop the rows.** If only a small fraction of rows have missing values and the dataset is large, simply removing those rows is the cleanest solution. If 0.5% of your data has a missing value in one column, dropping those rows loses almost nothing.

**Drop the column.** If a column is missing values for more than 30 to 40 percent of rows, it probably does not carry enough useful information to justify keeping it.

**Impute the values.** Fill in missing values with a reasonable estimate.

| Imputation Strategy | When to Use It |
|--------------------|----------------|
| Mean imputation | Numerical columns with no significant outliers |
| Median imputation | Numerical columns with outliers (median is more robust) |
| Mode imputation | Categorical columns (fill with the most frequent value) |
| Forward fill | Time series data where the previous value is a reasonable estimate |
| Model-based imputation | When you want a more accurate fill using other features |

```python
import pandas as pd

df['age'].fillna(df['age'].median(), inplace=True)
df['city'].fillna(df['city'].mode()[0], inplace=True)
```

### Duplicate Records

Duplicate rows can inflate your training data artificially and bias the model toward patterns that just happen to appear multiple times. Always check and remove them.

```python
df.drop_duplicates(inplace=True)
```

### Incorrect Data Types

A column storing numbers as strings, or dates stored as plain text, will cause errors during training. Type conversion is a basic but important cleaning step.

```python
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['date']  = pd.to_datetime(df['date'])
```

### Outliers

Outliers are data points that sit far outside the normal range. They can come from data entry errors (someone typed 1000 instead of 100), measurement errors, or genuine rare events.

**Detecting outliers:**

Using the IQR method — values below Q1 minus 1.5 times the IQR or above Q3 plus 1.5 times the IQR are considered outliers.

$$
\text{Lower bound} = Q1 - 1.5 \times IQR
$$
$$
\text{Upper bound} = Q3 + 1.5 \times IQR
$$

Using the Z-score method — values more than 3 standard deviations from the mean are outliers.

**Handling outliers:**

- Remove the row if it is clearly an error
- Cap the value at the boundary (called winsorizing)
- Keep it if it is a genuine rare event that the model should know about

There is no universal rule. Decide based on domain knowledge.

### Inconsistent Categories

Categorical columns often have the same value entered different ways: "New York", "new york", "NY", "N.Y." all mean the same thing but would be treated as four different categories.

```python
df['city'] = df['city'].str.strip().str.lower()
df['city'] = df['city'].replace({'ny': 'new york', 'n.y.': 'new york'})
```

---

## 2. Feature Engineering

Feature engineering is the process of using domain knowledge to create new features from the existing ones. The goal is to give the model better, more informative inputs so it can learn the pattern more easily.

A well-engineered feature can do more for model performance than switching to a more complex algorithm.

### Creating New Features from Existing Ones

**Date and time decomposition.** A raw timestamp carries hidden information that a model cannot directly use. Break it apart.

```python
df['hour']       = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month']      = df['timestamp'].dt.month
```

**Combining columns.** Sometimes the ratio or product of two features is more informative than either alone.

```python
df['price_per_sqft'] = df['price'] / df['area']
df['bmi']            = df['weight'] / (df['height'] ** 2)
```

**Binning continuous variables.** Grouping a continuous value into categories can sometimes help the model find patterns tied to ranges rather than exact values.

```python
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 60, 100],
                          labels=['teen', 'young_adult', 'adult', 'senior'])
```

**Interaction features.** Multiplying two features together to capture their combined effect.

```python
df['rooms_x_location'] = df['num_rooms'] * df['location_score']
```

### Encoding Categorical Variables

Machine learning models work with numbers. Categorical text columns must be converted.

**Label encoding:** Assign each category an integer. Simple but implies an ordering that may not exist.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])
```

**One-hot encoding:** Create a separate binary column for each category. No false ordering implied. Can increase dimensionality significantly for high-cardinality columns.

```python
df = pd.get_dummies(df, columns=['city'], drop_first=True)
```

**Ordinal encoding:** When categories have a natural order (low, medium, high), map them to ordered numbers.

```python
order_map = {'low': 0, 'medium': 1, 'high': 2}
df['priority'] = df['priority'].map(order_map)
```

---

## 3. Feature Scaling and Normalisation

Features in a dataset often have very different numerical ranges. House size might be in the thousands, number of rooms in the single digits, and location score between 0 and 10. Without scaling, algorithms that rely on distance or gradient magnitude will give far more weight to features with larger numbers simply because those numbers are bigger — not because those features are more important.

### Min-Max Normalisation

Scales all values to a fixed range, usually 0 to 1.

$$
x' = \frac{x - x_{min}}{x_{max} - x_{min}}
$$

All values are compressed into the 0 to 1 interval. Simple and interpretable. Sensitive to outliers because one extreme value will compress everything else into a tiny range.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['size', 'age']] = scaler.fit_transform(df[['size', 'age']])
```

### Standardisation (Z-Score Normalisation)

Scales values so the feature has mean zero and standard deviation one.

$$
x' = \frac{x - \mu}{\sigma}
$$

More robust to outliers than min-max. Output values are unbounded but most fall between -3 and 3. This is the default choice in most situations.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['size', 'age']] = scaler.fit_transform(df[['size', 'age']])
```

### The Critical Rule

Always fit the scaler on training data only, then use those same parameters to transform both training and test data. Fitting on test data would leak information about the test distribution into your pipeline.

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### Which Algorithms Need Scaling

Algorithms that compute distances or use gradient descent are sensitive to scale. Tree-based algorithms are not.

| Needs Scaling | Does Not Need Scaling |
|--------------|----------------------|
| Linear Regression | Decision Trees |
| Logistic Regression | Random Forest |
| Neural Networks | XGBoost / Gradient Boosting |
| KNN | Naive Bayes |
| SVM | |
| PCA | |

---

## 4. Dimensionality Reduction

Dimensionality reduction is the process of reducing the number of features while retaining as much useful information as possible. High-dimensional data (many features) causes several problems: models overfit more easily, training is slower, and visualisation becomes impossible. This collection of problems is often called the **curse of dimensionality**.

### Principal Component Analysis (PCA)

PCA is the most widely used dimensionality reduction technique. It finds new axes (called principal components) in the direction of maximum variance in the data, then projects the data onto a smaller number of these axes.

The principal components are linear combinations of the original features. The first component captures the most variance, the second captures the most remaining variance while being orthogonal to the first, and so on.

```
Original features: [size, rooms, age, location, floor, distance_to_metro, ...]
                             100 features
                                |
                               PCA
                                |
                    PC1, PC2, PC3, PC4, PC5
                             5 components
           (capturing 95% of the variance in the original data)
```

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)    # keep enough components to explain 95% of variance
X_reduced = pca.fit_transform(X_scaled)
```

**Important:** PCA requires scaled data. Run standardisation before PCA.

**Trade-off:** Principal components are not directly interpretable. You lose the ability to say "feature X has weight W". The components are abstract combinations of original features.

### t-SNE (t-Distributed Stochastic Neighbour Embedding)

t-SNE is used purely for **visualisation**, not for feeding into a model. It reduces high-dimensional data to 2 or 3 dimensions in a way that preserves local structure — data points that were close together in high-dimensional space tend to remain close in the 2D plot.

It is excellent for understanding how well your data clusters and whether classes are linearly separable. It is computationally expensive and non-deterministic, so the results change between runs.

```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_2d = tsne.fit_transform(X_scaled)
```

### When to Use Dimensionality Reduction

- You have more features than training examples (wide data)
- Training is too slow and you need to speed it up
- You want to visualise high-dimensional data
- Features are highly correlated with each other (multicollinearity)

---

## 5. Feature Selection

Feature selection is different from dimensionality reduction. Instead of creating new compressed features, you **select a subset of the original features** and discard the rest. The remaining features stay interpretable and meaningful.

Too many features cause overfitting, slow training, and add noise from irrelevant columns. Feature selection addresses all three.

### Filter Methods

These evaluate each feature independently based on a statistical measure, without involving the model at all. They are fast but do not account for interactions between features.

**Correlation with target:** Remove features with very low correlation to the output variable.

```python
correlation = df.corr()['target'].abs().sort_values(ascending=False)
# Keep features with correlation above a threshold
```

**Variance threshold:** Remove features that barely change across the dataset. A feature that is almost constant adds no information.

```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)
```

**Chi-squared test:** For categorical features and classification tasks, test statistical dependence between each feature and the target.

### Wrapper Methods

These use a model to evaluate feature subsets. They are more accurate than filter methods but computationally expensive.

**Recursive Feature Elimination (RFE):** Train the model on all features, rank features by importance, remove the least important one, retrain, repeat until the desired number of features is reached.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
```

### Embedded Methods

Feature selection happens as part of the model training process. The model itself learns which features matter.

**Lasso (L1 regularization):** As covered in regularization notes, L1 regularization pushes irrelevant feature weights to exactly zero. Features with zero weight are effectively selected out.

```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.01)
model.fit(X, y)
important_features = X.columns[model.coef_ != 0]
```

**Tree-based feature importance:** Decision trees and ensemble methods like Random Forest compute a feature importance score based on how much each feature reduces impurity across all splits.

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False)
```

### Filter vs Wrapper vs Embedded

| Method | Speed | Accuracy | Considers Interactions | Example |
|--------|-------|----------|----------------------|---------|
| Filter | Fast | Lower | No | Correlation, variance threshold |
| Wrapper | Slow | Higher | Yes | RFE |
| Embedded | Medium | High | Yes (indirectly) | Lasso, Random Forest importance |

---

## The Typical Preprocessing Pipeline

In practice, these steps follow a logical order. Doing them out of sequence causes problems.

```
Raw Data
    |
    v
Data Cleaning
(handle missing values, duplicates, outliers, type errors)
    |
    v
Feature Engineering
(create new columns, encode categoricals)
    |
    v
Feature Selection
(remove irrelevant or redundant features)
    |
    v
Feature Scaling
(standardise or normalise numerical columns)
    |
    v
Dimensionality Reduction (optional)
(PCA if still too many features or multicollinearity present)
    |
    v
Split into Train and Test sets
    |
    v
Ready for Model Training
```

Scaling and dimensionality reduction always come after splitting, and the scaler or PCA is always fit on training data only.

---

## Quick Concept Summary

| Concept | What It Means |
|---------|--------------|
| Data cleaning | Fixing missing values, duplicates, outliers, and type errors |
| Imputation | Filling missing values with mean, median, mode, or model predictions |
| Feature engineering | Creating new informative features from existing ones |
| One-hot encoding | Converting a categorical column into multiple binary columns |
| Label encoding | Mapping categories to integers |
| Min-max normalisation | Scaling values to the range 0 to 1 |
| Standardisation | Scaling to mean 0 and standard deviation 1 |
| Dimensionality reduction | Compressing many features into fewer while retaining most information |
| PCA | Projects data onto axes of maximum variance |
| Feature selection | Choosing the most useful original features and discarding the rest |
| Filter methods | Select features using statistics, no model involved |
| Wrapper methods | Select features by evaluating model performance on subsets |
| Embedded methods | Feature selection built into the model training process |
| Curse of dimensionality | Problems that arise from having too many features relative to data size |
