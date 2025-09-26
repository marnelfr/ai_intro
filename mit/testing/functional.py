from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Example values
m = 5   # input features
n1 = 8  # neurons in first hidden layer
n2 = 4  # neurons in second hidden layer
k = 2   # output classes

inputs  = keras.Input(shape=(m,), name="x")                       # input layer
h1      = layers.Dense(n1, activation="relu", name="L1")(inputs)  # hidden layer 1
h2      = layers.Dense(n2, activation="relu", name="L2")(h1)      # hidden layer 2
outputs = layers.Dense(k, activation="softmax", name="Out")(h2)   # output layer

model = keras.Model(inputs, outputs)
model.summary()

# -------------------------------
# Test the model with random input
# -------------------------------
x_sample = np.random.random((1, m))  # 1 sample with m features
y_pred = model(x_sample)
print("Input:", x_sample)
print("Predicted output:", y_pred)