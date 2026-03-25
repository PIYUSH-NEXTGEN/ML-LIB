# =========================
# IMPORTS
# =========================
# Import core libraries for data manipulation, visualization datasets, and numerical operations
import pandas as pd
import seaborn as sns
import numpy as np


# =========================
# DATA LOADING
# =========================
# Load datasets from CSV files into pandas DataFrames
nov = pd.read_csv('reg-month1.csv')
dec = pd.read_csv('reg-month2.csv')
students = pd.read_csv('students.csv')
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')


# =========================
# CONCATENATION
# =========================
# Combine multiple DataFrames either vertically (default) or horizontally (axis=1)
# print(pd.concat([nov, dec]))             # Vertical concat (recommended)
# print(pd.concat([nov, dec], axis=1))     # Horizontal concat
# print(nov._append(dec))                  # Deprecated method


# =========================
# MULTI-INDEX DATAFRAME
# =========================
# Create hierarchical indexing using keys to distinguish data sources
# multi = pd.concat([nov, dec], keys=['Nov', 'Dec'])
# print(multi)
# print(multi.loc[('Nov', 0)])
# print(multi.loc[('Dec', 5)])


# =========================
# MERGE / JOIN OPERATIONS
# =========================
# Combine DataFrames based on common columns using SQL-like joins
# print(students.merge(nov, how='inner', on='student_id'))
# print(students.merge(nov, how='left', on='student_id'))
# print(students.merge(nov, how='right', on='student_id'))
# print(students.merge(nov, how='outer', on='student_id'))


# =========================
# MULTI-INDEX CREATION
# =========================
# Generate hierarchical index manually using tuples or Cartesian product

# From tuples
# index_val = [
#     ('cse', 2019), ('cse', 2020), ('cse', 2021), ('cse', 2022),
#     ('cse', 2019), ('cse', 2020), ('cse', 2021), ('cse', 2022)
# ]
# multiindex = pd.MultiIndex.from_tuples(index_val)
# print(multiindex)

# From product
# print(pd.MultiIndex.from_product([['cse', 'ece'], [2019, 2020, 2021, 2022]]))


# =========================
# PIVOT TABLE
# =========================
# Summarize data into a 2D table by grouping and aggregating values
# df = sns.load_dataset('tips')
# print(df)

# print(df.pivot_table(index='sex', columns='smoker', values='total_bill'))                  # Default mean
# print(df.pivot_table(index='sex', columns='smoker', values='total_bill', aggfunc='sum'))   # Custom aggregation


# =========================
# STRING OPERATIONS
# =========================
# Apply vectorized string methods efficiently on pandas Series
# s = pd.Series(['cat', 'mat', None])
# print(s.str.startswith('c'))

# titanic = pd.read_csv('titanic.csv')
# print(titanic['Name'].str.upper())
# print(titanic['Name'].str.lower())
# print(titanic['Name'].str.title())
# print(titanic['Name'].str.len().max())
# print(titanic['Name'][titanic['Name'].str.len() == 82].values[0])


# =========================
# DATE & TIME
# =========================
# Work with timestamps and extract date-time components
# print(pd.Timestamp('2026/3/26'))

# x = pd.Timestamp('2026-03-26 01:05')
# print(x.year, x.month, x.day, x.hour)

# NumPy datetime
# date = np.array('2026-03-26', dtype='datetime64[D]')
# print(date)


# =========================
# DATE RANGE GENERATION
# =========================
# Generate sequences of dates using a fixed range or number of periods
# print(pd.date_range(start='2026-03-26', end='2026-04-01', freq='D'))
# print(pd.date_range(start='2026-03-26', periods=25, freq='D'))


# =========================
# TO_DATETIME
# =========================
# Convert strings to datetime format and handle invalid parsing cases
# c = pd.Series(['2020-1-1', '2021-2-2'])
# print(pd.to_datetime(c))
# print(pd.to_datetime(c).dt.year)

# d = pd.Series(['2020-1-1', '2021-200-2'])
# print(pd.to_datetime(d, errors="coerce"))   # Invalid dates become NaT

