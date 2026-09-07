import numpy as np

inputs = np.array([5, 8])
print("Input:", inputs)

# 3 neurons in the hidden layer. Each row represents the weights of one neuron.
weights = np.array([
    [0.5, 0.2],   # Neuron 1
    [0.8, 0.4],   # Neuron 2
    [0.3, 0.9]    # Neuron 3
])
# One bias for each neuron
bias = np.array([0.1, 0.2, 0.3])

# sigmoid func
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# z = wx + b
z_hidden = np.dot(weights, inputs) + bias
print(f"\nHidden Layer Z: {z_hidden}")

# Apply activation function
output = sigmoid(z_hidden)
print(f"\nHidden Layer Output: {output}")

# Hidden layer has 3 outputs, therefore output neuron has 3 weights.
weights_output = np.array([0.7, 0.5, 0.2])
bias_output = 0.1
z_output = np.dot(weights_output,output) + bias_output
print(f"\nOutput Layer Z: {z_output}")

prediction = sigmoid(z_output)
print(f"\nFinal Prediction: {prediction}")


# # WITH TENSORFLOW
# from tensorflow.keras.models import Sequential      # A Sequential model means layers are added one after another.
# from tensorflow.keras.layers import Dense           # Dense = Fully Connected Layer, Every neuron connects to every neuron in the previous layer.
#
# # Column 1 = Hours Studied
# # Column 2 = Hours Slept
# X = np.array([
#     [5, 8],
#     [2, 5],
#     [8, 9],
#     [1, 3],
#     [6, 8],
#     [3, 4]
# ])
# y = np.array([
#     1,
#     0,
#     1,
#     0,
#     1,
#     0
# ])
# model = Sequential()            # Creates an empty Sequential model. Layers will be added one by one.
#
# # Dense Layer
# model.add(
#     Dense(
#         units=3,                # units = 3 -> Create 3 neurons in this hidden layer.
#         activation='sigmoid',   # activation = 'sigmoid' -> Apply the sigmoid activation function to each neuron.
#         input_shape=(2,)        # input_shape = (2,) -> Each training example has 2 input features
#     )
# )
# # Create the output layer.
# model.add(
#     Dense(
#         units=1,                # units = 1 -> One neuron because we are predicting one value (Pass or Fail).
#         activation='sigmoid'    # activation = 'sigmoid' -> Output will be between 0 and 1 (probability).
#     )
# )
# model.summary()
# model.compile(
#     optimizer='adam',           # optimizer='adam'  -> Algorithm used to update weights and biases.
#     loss='binary_crossentropy', # loss='binary_crossentropy' -> Cost function used for binary classification.
#     metrics=['accuracy']        # metrics=['accuracy'] -> Displays accuracy after every epoch.
# )
# model.fit(
#     X,
#     y,
#     epochs=100                 # epochs=100 means the entire dataset is used 100 times for training.
# )
# new_student = np.array([[1, 3 ]])
# prediction = model.predict(new_student)
# print((prediction > 0.5).astype(int))












