import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# =========================
# DATA LOADING
# =========================
# Load dataset for categorical and distribution-based analysis
tips = sns.load_dataset('tips')


# =========================
# STRIP PLOT (DISTRIBUTION WITH POINTS)
# =========================
# Show distribution of values across categories using individual data points

# sns.stripplot(
#     data=tips,
#     x='day',
#     y='total_bill',
#     hue='sex',
#     jitter=False     # controls overlap of points
# )


# =========================
# BOX PLOT (SUMMARY STATISTICS)
# =========================
# Visualize median, quartiles, and outliers for each category

# sns.boxplot(
#     data=tips,
#     x='day',
#     y='total_bill'
# )


# =========================
# VIOLIN PLOT (DISTRIBUTION + DENSITY)
# =========================
# Combine boxplot with KDE to show full distribution shape

# sns.violinplot(
#     data=tips,
#     x='day',
#     y='total_bill',
#     hue='sex'
# )

# =========================
# COUNT PLOT (CATEGORICAL FREQUENCY)
# =========================
# Count occurrences of categories and compare distributions

# sns.countplot(
#     data=tips,
#     x='sex',
#     hue='day'
# )

# =========================
# REGRESSION PLOT (TREND LINE)
# =========================
# Show relationship between variables with best-fit regression line

# sns.regplot(
#     data=tips,
#     x='total_bill',
#     y='tip'
# )

# =========================
# FACET GRID (MULTI-DIMENSIONAL SPLIT)
# =========================
# Create grid of plots across multiple categorical variables

# x = sns.FacetGrid(data=tips, col='day', row='time')
# x.map(sns.violinplot, 'sex', 'total_bill')

# =========================
# PAIR PLOT (FEATURE RELATIONSHIPS)
# =========================
# Visualize pairwise relationships between all numerical features

iris = px.data.iris()

# sns.pairplot(data=iris, hue='species')

# =========================
# PAIR GRID (CUSTOM MULTI-PLOT MATRIX)
# =========================
# Build customizable pairwise plots across multiple variables

k = sns.PairGrid(data=iris)
k.map(sns.scatterplot)

plt.show()

