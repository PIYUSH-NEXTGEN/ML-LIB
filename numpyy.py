"""
Topic: NumPy Fundamentals Practice
Goal: Exploring array creation, manipulation, statistics, reshaping, stacking,
splitting, indexing, and type conversion.
"""

import numpy as np   # importing NumPy library (np is the standard alias)

# ----------------------------------------
# 1. Array Creation Methods
# ----------------------------------------

# arr = np.array([1,2,3,4])   # np.array() → creates a NumPy array from a list or tuple
# print(arr)

# arr2 = np.array([[1,2,3,],[3,4,5,]])   # creates a 2D array (matrix)
# print(arr2)

# arr3 = np.zeros(4)   # np.zeros() → creates array filled with 0s
# print(arr3)

# arr4 = np.ones((2,3))   # np.ones() → creates array filled with 1s
# print(arr4)

# arr5 = np.random.random((3,5))   # np.random.random() → creates array with random values between 0 and 1
# print(arr5)

# arr6 = np.full((2,4),12)   # np.full() → creates array filled with a specific value
# print(arr6)

# arr7 = np.arange(1,10,3)   # np.arange() → creates array with start, stop, step
# print(arr7)

# arr8 = np.eye(4)   # np.eye() → creates identity matrix
# print(arr8)

# ----------------------------------------
# 2. Deleting, Inserting, Appending
# ----------------------------------------

# arr9 = np.array([[1,2,3],
#                  [4,5,6],
#                  [7,8,9]])
# arr_dlt = np.delete(arr9,1,0)   # np.delete() → deletes row/column (here deletes row index 1)
# print(arr_dlt)

# arr10 = np.array([7,2,53,3,4,1])
# arr_insert = np.insert(arr10,5,6)   # np.insert() → inserts value at given index
# arr_insert = np.insert(arr9,3,[10,11,12],axis=0)
# print(arr_insert)

# arr_append = np.append(arr10,[10,20])   # np.append() → adds elements at end
# print(arr_append)

# ----------------------------------------
# 3. Array Properties and Statistics
# ----------------------------------------

# print(arr9.shape)   # .shape → returns dimensions of array
# print(arr9.size)    # .size → total number of elements
# print(arr9.ndim)    # .ndim → number of dimensions
# print(arr9.dtype)   # .dtype → data type of elements
# print(np.sort(arr10))   # np.sort() → sorts array
# print(arr10.astype(str))   # .astype() → converts datatype
# print(np.mean(arr10))   # np.mean() → average of elements
# print(np.var(arr10))    # np.var() → variance of elements
# print(np.sum(arr10))    # np.sum() → sum of elements

# ----------------------------------------
# 4. Reshaping and Flattening
# ----------------------------------------

# arr11 = np.array([7,2,9,3,4,1])
# print(arr11.reshape(2,3))   # .reshape() → changes shape without changing data

# arr12 = np.array([[1,2,3],[4,5,6]])
# print(arr12.ravel())   # .ravel() → flattens array (returns view if possible)
# print(arr12.flatten()) # .flatten() → flattens array (returns copy)

# ----------------------------------------
# 5. Stacking and Splitting
# ----------------------------------------

# arr13 = np.array([1,2,3])
# arr14 = np.array([4,5,6])
# arr15 = np.array([7,8,9])

# print(np.vstack((arr10,arr11)))   # np.vstack() → stacks arrays vertically (row-wise)
# print(np.hstack((arr13,arr14,arr15)))   # np.hstack() → stacks arrays horizontally

# arr16 = np.array([10,30,85,82])
# arr17 = np.array([[10,30,85,82],[94,62,97,8]])
# print(np.split(arr16,2))   # np.split() → splits array into equal parts
# print(np.vsplit(arr17,2))  # np.vsplit() → splits array vertically (row-wise)

# ----------------------------------------
# 6. Concatenation and Sequence Generation
# ----------------------------------------

# arr18 = np.array([1,2,3])
# arr19 = np.array([4,5,6])
#
# arr20 = np.concatenate((arr18,arr19),axis=0)   # np.concatenate() → joins arrays along axis
# print(arr20)

# arr21 = np.linspace(10,2,20)   # np.linspace() → creates evenly spaced numbers between start & stop
# print(arr21)

# ----------------------------------------
# 7. Indexing and Filtering
# ----------------------------------------

# print(arr[1:5])   # slicing → elements from index 1 to 4
# print(arr[:5])
# print(arr[::2])   # step slicing
# print(arr[::-1])  # reverse array

# fancy indexing → selecting multiple elements using list of indices
# print(arr[[0,2,4]])

# boolean masking → filter elements based on condition
# print(arr[arr>2])

# ----------------------------------------
# 8. Sorting and Type Conversion (Additional)
# ----------------------------------------

# arr22 = np.array([4,5,7,2,9,8])
# print(np.sort(arr22))

# arr23 = np.array([1,23,4])
# print(arr23.astype(str))
