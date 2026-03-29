import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# =========================
# DATA LOADING (SEABORN)
# =========================
# Load built-in dataset for basic visualization examples
tips = sns.load_dataset('tips')


# =========================
# SCATTER PLOT
# =========================
# Visualize relationship between total bill and tip (axis-level function)

# sns.scatterplot(data=tips, x='total_bill', y='tip')


# =========================
# RELATIONAL PLOT (FIGURE-LEVEL)
# =========================
# Create advanced scatter plots with grouping, styling, and sizing

# sns.relplot(
#     data=tips,
#     x='total_bill',
#     y='tip',
#     kind='scatter',
#     hue='sex',        # color grouping
#     style='time',     # marker style variation
#     size='size'       # size encoding
# )


# =========================
# LINE PLOT (TIME SERIES)
# =========================
# Analyze trends over time using country-level data

gap = px.data.gapminder()

# df = gap[gap['country'] == 'India']
# sns.lineplot(data=df, x='year', y='lifeExp')

# Using relplot (figure-level alternative)
# sns.relplot(data=df, x='year', y='lifeExp', kind='line')

# Multi-country comparison
# temp_df = gap[gap['country'].isin(['India','Pakistan','China'])]
# sns.relplot(data=temp_df, x='year', y='lifeExp', kind='line', hue='country')


# =========================
# FACET GRID (MULTIPLE SUBPLOTS)
# =========================
# Split data into multiple panels based on categorical variables

# sns.relplot(
#     data=tips,
#     x='total_bill',
#     y='tip',
#     kind='scatter',
#     col='smoker',
#     row='sex'
# )


# =========================
# COLUMN WRAPPING
# =========================
# Create multiple plots across categories (like time progression)

# sns.relplot(
#     data=gap,
#     x='lifeExp',
#     y='gdpPercap',
#     kind='scatter',
#     col='year'
# )


# =========================
# DISTRIBUTION PLOTS
# =========================
# Analyze distribution of a single variable (univariate analysis)

# sns.histplot(data=tips, x='total_bill')                       # histogram

# sns.displot(data=tips, x='total_bill', kind='hist')           # figure-level histogram
# sns.displot(data=tips, x='tip', kind='hist', hue='sex')       # grouped histogram
# sns.displot(data=tips, x='tip', kind='hist', col='sex', element='step')


# =========================
# KDE PLOT (DENSITY ESTIMATION)
# =========================
# Estimate probability density of continuous variables

# sns.kdeplot(data=tips, x='total_bill')
# sns.displot(data=tips, x='total_bill', kind='kde', hue='sex')


# =========================
# RUG PLOT
# =========================
# Show individual data points along axis for distribution insight

# sns.rugplot(data=tips, x='total_bill')


# =========================
# BIVARIATE DISTRIBUTION
# =========================
# Analyze joint distribution of two variables

# sns.histplot(data=tips, x='total_bill', y='tip')              # 2D histogram
# sns.kdeplot(data=tips, x='total_bill', y='tip')               # 2D density


# =========================
# HEATMAP (MATRIX VISUALIZATION)
# =========================
# Convert data into matrix form and visualize intensity patterns

# df_2 = gap.pivot(index='country', columns='year', values='lifeExp')
# plt.figure(figsize=(15,15))
# sns.heatmap(df_2, annot=True, linewidths=0.5, cmap='viridis')


# =========================
# CLUSTER MAP
# =========================
# Perform hierarchical clustering and visualize grouped similarity

iris = px.data.iris()
sns.clustermap(iris.iloc[:, [0,1,2,3]])

plt.show()