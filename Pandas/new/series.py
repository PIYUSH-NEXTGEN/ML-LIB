import pandas as pd
import numpy as np

# SERIES FROM LIST
# country = ['India','Srilanka','China','Japan']
# print(pd.Series(country))

# batsman = ["rohit","kohli","dhoni","buttler"]
# runs = [10,20,58,83]
# print(pd.Series(runs))

 # custom indexing
# print(pd.Series(runs,index=batsman))

  # setting a name
# print(pd.Series(runs,index=batsman,name="Cricketers runs"))

# SERIES FROM DICT
# runs = {
#     "rohit":10,
#     "kohli":20,
#     "dhoni":58,
#     "buttler":83
# }
# stats = pd.Series(runs)
# print(stats)

#attributes -
# print(stats.dtype)
# print(stats.size)
# print(stats.name)
# print(stats.is_unique)
# print(stats.index)
# print(stats.values)

# series using read csv -
file = pd.read_csv("subs.csv")
# print(file)

file_2  = pd.read_csv("kohli_ipl.csv",index_col='match_no')
# print(file_2)

file_3 = pd.read_csv("bollywood.csv",index_col='movie')
# print(file_3)

# SERIES METHOD -

 # head -> print first 5 rows by default
# print(file.head())
# print(file.head(4)) # -> only for first 4

  # tail -> opp of head

  # sample -> give one row randomly
# print(file.sample())
# print(file.sample(2))

  # value_count
# print(file_3.value_counts())

 # sort_values
# print(file_2.sort_values("runs"))
# print(file_2.sort_values("runs",ascending=False))
# print(file_2.sort_values("run s",ascending=False).head(1).values[0])

 # sort_index
# print(file_3.sort_index())

 # count
# print(file_2.count())

 # sum  -> product
# print(file.sum())

 # mean,mode,median,var,std
# print(file.mean())
# print(file.mode())
# print(file.median())
# print(file.std())
# print(file.var())

 # min max
# print(file_2.max())
# print(file_2.min())
# print(file_2.describe())

 # slicing
# print(file_2[2:10])

#arithmetic op
# print(100 - file_2)

# relational op
# print(file_2 >= 50)









