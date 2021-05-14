import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.python.keras.applications.vgg19 import preprocess_input 

# print(tf.__version__) # Have to use at least tf 2.3

img_height = img_width = 224
batch_size = 15

# Use dataset from directory
directory = "./datasets/six_shapes_t/"

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    directory, 
    labels = "inferred",
    label_mode = "int",
    color_mode = "rgb",
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 3,
    validation_split = 0.1,
    subset = "training"
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    directory, 
    labels = "inferred",
    label_mode = "int",
    color_mode = "rgb",
    batch_size = batch_size,
    image_size = (img_height, img_width),
    shuffle = True,
    seed = 3,
    validation_split = 0.1,
    subset = "validation"
)

class_names = ds_train.class_names 
# print(class_names)

num_classes = len(class_names)

# Transfer Learning w/ VGG Model (using pretrained model as feature extractor)

preprocess_input = keras.applications.vgg19.preprocess_input

vgg_model = keras.applications.VGG19(input_shape = [img_height,img_width,3], weights='imagenet', include_top=False)
vgg_model.trainable = False
# vgg_model.summary()

inputs = keras.Input(shape=(img_height,img_width,3))
x = preprocess_input(inputs)
x = vgg_model(x, training = False)

# Custom Classification Head (Explore other possibilities)
x = layers.Flatten()(x)
outputs = layers.Dense(num_classes, activation = "softmax")(x)

model = keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

num_epochs = 20
history = model.fit(
  ds_train,
  epochs=num_epochs,
  verbose=2,
  validation_data=ds_validation
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(num_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Testing Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Testing Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Testing Loss')
plt.legend(loc='upper right')
plt.title('Training and Testing Loss')
plt.show()
# plt.savefig("loss_accuracy_plots.png", bbox_inches="tight")