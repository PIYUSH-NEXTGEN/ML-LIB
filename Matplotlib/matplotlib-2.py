import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================
# DATA LOADING (IRIS)
# =========================
# Load Iris dataset for feature relationship visualization
iris = pd.read_csv('iris.csv')


# =========================
# SCATTER PLOT (IRIS DATA)
# =========================
# Visualize relationship between sepal length and petal length with species-based coloring

# iris['Species'] = iris['Species'].replace({
#     'Iris-setosa': 0,
#     'Iris-versicolor':
#     'Iris-virginica': 2
# })

# print(iris['Species'].dtype)

# plt.scatter(
#     iris['SepalLengthCm'],
#     iris['PetalLengthCm'],
#     c=iris['Species']
# )

# plt.xlabel('Sepal length')
# plt.ylabel('Petal length')
# plt.colorbar()                         # show mapping of numeric values to colors
# plt.figure(figsize=(10,5))              # adjust figure size


# =========================
# DATA LOADING (BATTER)
# =========================
# Load IPL batter dataset for performance analysis
batter = pd.read_csv('batter.csv')


# =========================
# SCATTER PLOT (FILTERED SAMPLE)
# =========================
# Plot sampled data points with bubble size representing runs

# sample_df = batter.head(100).sample(25, random_state=5)
# plt.figure(figsize=(15,7))
# plt.scatter(sample_df['avg'], sample_df['strike_rate'], s=sample_df['runs'])

# plt.axhline(130, color='red')           # horizontal reference line
# plt.axvline(30, color='red')            # vertical reference line


# =========================
# SUBPLOTS (MULTIPLE CHARTS)
# =========================
# Create multiple plots within a single figure for comparative analysis

# plt.scatter(batter['avg'], batter['strike_rate'])

# fig, ax = plt.subplots()
# ax.scatter(batter['avg'], batter['strike_rate'])

# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,8))

# ax[0,0].scatter(batter['avg'], batter['strike_rate'])
# ax[0,1].scatter(batter['avg'], batter['runs'])
# ax[1,0].hist(batter['avg'])
# ax[1,1].hist(batter['runs'])


# =========================
# 3D SCATTER PLOT
# =========================
# Visualize three variables simultaneously in 3D space

# fig = plt.figure()
# ax = plt.subplot(projection='3d')
# ax.scatter(batter['runs'], batter['avg'], batter['strike_rate'])


# =========================
# 3D SURFACE PLOT
# =========================
# Generate a surface plot to visualize mathematical function in 3D

# x = np.linspace(-10, 10, 100)
# y = np.linspace(-10, 10, 100)

# xx, yy = np.meshgrid(x, y)              # create coordinate grid

# z = xx**2 + yy**2                       # define surface function

# ax = plt.subplot(projection='3d')
# p = ax.plot_surface(xx, yy, z)


# =========================
# HEATMAP (MATCH ANALYSIS)
# =========================
# Show frequency of sixes across overs and ball numbers using a grid

# delivery = pd.read_csv('IPL_Ball_by_Ball_2008_2022.csv')
# plt.figure(figsize=(8,8))

# temp_df = delivery[
#     (delivery['ballnumber'].isin([1,2,3,4,5,6])) &
#     (delivery['batsman_run'] == 6)
# ]

# grid = temp_df.pivot_table(
#     index='overs',
#     columns='ballnumber',
#     values='batsman_run',
#     aggfunc='count'
# )

# plt.imshow(grid)                        # display matrix as image
# plt.colorbar()                          # show intensity scale

plt.show()