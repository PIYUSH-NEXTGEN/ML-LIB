import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# =========================
# BASIC LINE PLOT
# =========================
# Plot a simple line graph using manually defined lists

# price = [50,100,200,500,700,300]
# year = [2015,2016,2017,2018,2019,2020]

# plt.plot(year, price)   # x-axis = year, y-axis = price
# plt.show()              # render the plot


# =========================
# DATA LOADING (CSV)
# =========================
# Load dataset containing player performance data
batsman = pd.read_csv("sharma-kohli.csv")


# =========================
# LINE PLOT (REAL DATA)
# =========================
# Compare multiple players using line plots on same graph

# plt.plot(batsman['index'], batsman["V Kohli"], color='red', linestyle="dashed", label='Virat')
# plt.plot(batsman['index'], batsman["RG Sharma"], color='blue', linestyle="dotted", linewidth=5, label='Rohit')

# Customize appearance
# plt.title('Rohit Sharma vs Virat Kohli')   # title of the graph
# plt.xlabel('Season')                       # x-axis label
# plt.ylabel('Runs')                         # y-axis label
# plt.legend()                               # display legend for multiple lines
# plt.grid()                                 # add grid for better readability
# plt.show()


# =========================
# SCATTER PLOT
# =========================
# Generate random data and visualize relationship between two variables

# x = np.linspace(-10, 10, 50)                       # evenly spaced values
# y = 10*x + 3 + np.random.randint(0, 300, 50)       # linear relation + noise
# plt.scatter(x, y)                                  # scatter plot
# plt.show()


# =========================
# SCATTER PLOT
# =========================
# Visualize relationship between batting average and strike rate

# df = pd.read_csv("batter.csv")
# df = df.head(50)                                   # select top 50 rows
# plt.scatter(df['avg'], df['strike_rate'])
# plt.title("Top 50 batsman in IPL")
# plt.xlabel("Average")
# plt.ylabel("Strike rate")
# plt.grid()


# =========================
# SCATTER (SEABORN DATASET)
# =========================
# Use built-in dataset to visualize bill vs tip with size variation

# tips = sns.load_dataset('tips')
# plt.scatter(tips['total_bill'], tips['tip'], s=tips['size']*20)


# =========================
# BAR GRAPH (BASIC)
# =========================
# Display categorical data using bar charts

# children = [10,20,40,60,30]
# colors = ['red','blue','green','black','pink']

# plt.bar(colors, children, color='black')    # vertical bar chart
# plt.bar(colors, children, color='black')    # horizontal comment incorrect but kept as-is


# =========================
# DATA LOADING (SEASON RECORD)
# =========================
# Load batsman performance across multiple seasons
batsman_rec = pd.read_csv('batsman_season_record.csv')


# =========================
# GROUPED BAR CHART
# =========================
# Plot side-by-side comparison for multiple years

# plt.bar(np.arange(batsman_rec.shape[0]) - 0.2, batsman_rec['2015'], width=0.2)
# plt.bar(np.arange(batsman_rec.shape[0]), batsman_rec['2016'], width=0.2)
# plt.bar(np.arange(batsman_rec.shape[0]) + 0.2, batsman_rec['2017'], width=0.2)

# plt.xticks(np.arange(batsman_rec.shape[0]), batsman_rec['batsman'])   # label x-axis


# =========================
# HISTOGRAM
# =========================
# Show frequency distribution of numerical data

# data = [10,20,30,50,70,60]

# plt.hist(data)                               # default bins
# plt.hist(data, bins=[10,40,60,80])            # custom bin ranges


# =========================
# PIE CHART
# =========================
# Represent proportions of categories as percentage slices

# data2 = [57,85,24,67,33]
# subjects_marks = ['maths','sci','eng','hindi','arts']

# plt.pie(data2, labels=subjects_marks, autopct='%0.1f%%', explode=[0.5,0,0,0,0])
plt.show()