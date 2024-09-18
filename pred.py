import silence_tensorflow.auto
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras



# Load and normalize MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# # Build the model
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')  # Use 'softmax' for classification
# ])

# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model
# model.fit(x_train, y_train, epochs=5)

# # Evaluate the model
# val_loss, val_acc = model.evaluate(x_test, y_test)
# print("Validation loss:", val_loss)
# print("Validation accuracy:", val_acc)

# model.save('nn_model.h5')

# with suppress_prints():
new_model = keras.models.load_model('nn_model.h5')
predictions = new_model.predict(x_test)
# print(predictions[0])
# print(predictions[0])
# print(predictions[0])
# print("First prediction: ", np.argmax(predictions[0]))
