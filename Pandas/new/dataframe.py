# =========================
# IMPORTS
# =========================
# Import pandas for data manipulation
import pandas as pd


# =========================
# DATA CREATION (CUSTOM)
# =========================
# Create a DataFrame manually using list data
stu_data = [
    [100,80,10,'piyush'],
    [38,19,60,'moksh'],
    [20,61,32,'zenos'],
    [12,65,76,'pac']
]

students = pd.DataFrame(stu_data, columns=['iq','marks','package','name'])
students.set_index('name', inplace=True)


# =========================
# DATA LOADING
# =========================
# Load datasets for movies and IPL matches
movies = pd.read_csv("movies.csv")
ipl = pd.read_csv("ipl-matches.csv")


# =========================
# BASIC DATA INSPECTION
# =========================
# Get overall structure, size, and metadata of DataFrames

# print(movies.shape)
# print(ipl.shape)

# print(movies.dtypes)
# print(ipl.dtypes)

# print(movies.index)
# print(ipl.index)

# print(movies.columns)
# print(ipl.columns)

# print(movies.values)
# print(ipl.values)

# print(movies.info)
# print(ipl.info)

# print(movies.describe())
# print(ipl.describe())


# =========================
# MISSING VALUES & DUPLICATES
# =========================
# Detect null values and duplicate rows in datasets

# print(movies.isnull)
# print(ipl.isnull)

# print(movies.duplicated())
# print(movies.duplicated().sum())


# =========================
# DATA MODIFICATION
# =========================
# Rename columns, compute aggregates, and transform data

# print(students.rename(columns={'marks':'percent'}))

# print(movies.sum)

# print(students.mean(axis=1))
# print(students.median(axis=1))
# print(students.max(axis=1))


# =========================
# COLUMN SELECTION
# =========================
# Select single or multiple columns from DataFrames

# print(movies[['title_x','year_of_release']])
# print(ipl[['Team1','Team2','Team3']])


# =========================
# INDEXING (ILOC & LOC)
# =========================
# Access data using index positions (iloc) or labels (loc)

# print(movies.iloc[4])
# print(movies.iloc[0:4])
# print(movies.iloc[0:20:2])

# print(movies.iloc[[0,4,5]])

# print(students.loc['piyush'])
# print(students.loc['piyush':'zenos'])

# print(students.iloc[0])

# print(movies.iloc[0:4, 0:3])


# =========================
# DATA FILTERING
# =========================
# Apply conditions to extract specific subsets of data

# Final match winners
# mask = ipl['MatchNumber'] == 'Final'
# new_df = ipl[mask]
# print(new_df[['Season','WinningTeam']])

# Super Over count
# so = ipl[ipl['SuperOver'] == 'Y']
# print(so.shape[0])
# print(so)

# CSK wins in Kolkata
# csk = ipl[(ipl['City'] == 'Kolkata') & (ipl['WinningTeam'] == 'Chennai Super Kings')]
# print(csk.shape[0])

# Toss winner equals match winner percentage
# toss_win = ipl[ipl['TossWinner'] == ipl['WinningTeam']].shape[0] / ipl.shape[0] * 100
# print(f"{toss_win:.2f}")

# Movies with rating > 8 and votes > 10000
# mov = movies['imdb_rating'] > 8 & (movies['imdb_votes'] > 10000)
# print(movies[mov])
# print(mov.shape[0])

# Action movies with rating > 7.5
# act_mov = movies['genres'].str.contains('Action')
# act_mov2 = movies['imdb_rating'] > 7.5
# print(movies[act_mov & act_mov2].shape[0])


# =========================
# FEATURE ENGINEERING
# =========================
# Add or modify columns in the dataset

# movies['Country'] = 'India'


# =========================
# TYPE CONVERSION
# =========================
# Convert column data types for memory optimization and efficiency

# ipl['Season'] = ipl['Season'].astype('category')
# ipl['Team1'] = ipl['Team1'].astype('category')
# ipl['Team2'] = ipl['Team2'].astype('category')
#
# print(ipl.info())





