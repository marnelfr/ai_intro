from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

inputs = keras.Input(shape=(2,), name="x")
h1 = layers.Dense(3, activation="relu", name="L1")(inputs)
outputs = layers.Dense(1, activation="sigmoid", name="Output")(h1)

model = keras.Model(inputs, outputs)
model.summary()

x_sample = np.random.random((10, 2))
y_pred = model(x_sample)
print("Input:", x_sample)
print("Prediction output:", y_pred)