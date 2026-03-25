# =========================
# IMPORTS
# =========================
# Import libraries for data manipulation and numerical operations
import pandas as pd
import numpy as np


# =========================
# DATA LOADING
# =========================
# Load datasets for IPL matches and movies analysis
ipl = pd.read_csv("ipl-matches.csv")
movies = pd.read_csv("movies.csv")


# =========================
# VALUE COUNTS
# =========================
# Count frequency of unique rows or values in Series/DataFrame
# marks = pd.DataFrame([
#     [100,80,10],
#     [90,70,7],
#     [120,100,14],
#     [80,70,14],
#     [80,70,14]
# ],columns=['iq','marks','package'])

# print(marks.value_counts())


# =========================
# IPL ANALYSIS
# =========================
# Perform basic analysis on IPL dataset

# Find player with most Player of the Match in finals/qualifiers
# matches = ipl[~ipl['MatchNumber'].str.isdigit()]
# print(matches['Player_of_Match'].value_counts())

# Toss decision distribution
# print(ipl['TossDecision'].value_counts())

# Total matches played by each team
# most_matches = (ipl['Team2'].value_counts() + ipl['Team1'].value_counts())
# print(most_matches.sort_values(ascending=False))


# =========================
# SORTING
# =========================
# Sort DataFrame values by column or index
# print(movies.sort_values('title_x'))


# =========================
# INDEX OPERATIONS
# =========================
# Set, reset, and modify DataFrame index

# print(ipl.set_index('ID'))
# ipl.set_index('ID', inplace=True)

# Replace index without losing it
# print(ipl.reset_index().set_index('ID'))


# =========================
# RENAME COLUMNS
# =========================
# Rename columns for better readability or consistency
# print(movies.rename(columns={'imdb_id':'imdb'}, inplace=True))


# =========================
# MISSING VALUES HANDLING
# =========================
# Detect and handle missing (NaN) values in data

# print(movies['imdb_id'].isnull())
# print(movies.isnull())

# print(movies['imdb_id'].notnull())

# print(movies['poster_path'].hasnans)

# Remove rows with missing values
# print(movies.dropna().size)
# print(movies.size)

# Fill missing values with a placeholder
# print(ipl.fillna('unknows'))


# =========================
# DUPLICATES
# =========================
# Identify duplicate rows in data

# print(marks.duplicated())
# print(marks.duplicated().size)


# =========================
# DROP OPERATIONS
# =========================
# Remove rows or columns from Series/DataFrame

# temp = pd.Series([10,2,3,16,3])
# print(temp.drop(index=[0,3]))

# print(movies.drop(columns=['imdb_id']))
# print(movies.drop(index=[0,4]))


# =========================
# GROUPBY OPERATIONS
# =========================
# Group data based on a column and perform aggregations
genre = movies.groupby('genres')

# print(genre.sum())
# print(genre.min())
# print(genre.mean)
# print(len(genre))
# print(genre.size())
# print(genre.value_counts())
# print(genre.get_group('Horror'))
# print(genre.groups)
# print(genre.describe())


# =========================
# APPLY FUNCTION
# =========================
# Apply custom function to each group in groupby object

def foo(group):
    print(group)
    return group

print(genre.apply(foo))