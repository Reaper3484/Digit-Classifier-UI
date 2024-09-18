import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

# Load and normalize MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Display a 5x5 grid of test images
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# Flatten the 2D grid of subplots into a 1D array
axes = axes.ravel()

# Loop through and display the first 25 images
for i in range(25):
    axes[i].imshow(x_test[i], cmap='gray')  # Show each image in grayscale
    axes[i].set_title(f"Label: {y_test[i]}")  # Show the label as the title
    axes[i].axis('off')  # Hide the axes

plt.tight_layout()
plt.show()
