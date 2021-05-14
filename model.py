import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 

# print(tf.__version__) # Have to use at least tf 2.3

img_height = img_width = 200
batch_size = 15

# Use dataset from directory
directory = "./datasets/three_shapes_size_t/"

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    directory, 
    labels = "inferred",
    label_mode = "int",
    color_mode = "grayscale",
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 33,
    validation_split = 0.2,
    subset = "training"
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    directory, 
    labels = "inferred",
    label_mode = "int",
    color_mode = "grayscale",
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 33,
    validation_split = 0.2,
    subset = "validation"
)

class_names = ds_train.class_names 
# print(class_names)

# Visualization Code
# plt.figure(figsize=(10, 10))
# for images, labels in ds_train.take(1):
#   for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(images[i].numpy().astype("uint8"),cmap="gray")
#     plt.title(class_names[labels[i]])
#     plt.axis("off")

# plt.show()

# num_classes = len(class_names)

# # Custom Model
# model = keras.Sequential([
#   layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dropout(0.5),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(64, activation="relu"),
#   layers.Dense(num_classes, activation="softmax")
# ])

# model.compile(optimizer=keras.optimizers.Adam(),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# model.summary()

# num_epochs = 15
# history = model.fit(
#   ds_train,
#   epochs=num_epochs,
#   verbose=2,
#   validation_data=ds_validation
# )

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(num_epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Testing Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Testing Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Testing Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Testing Loss')
# plt.savefig("loss_accuracy_plots_large_0.png", bbox_inches="tight")
