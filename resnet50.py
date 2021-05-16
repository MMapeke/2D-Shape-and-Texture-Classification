from logging import error
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL

import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers

# print(tf.__version__) # Have to use at least tf 2.4

# ADDING CHECKPOINT FOR MODEL
checkpoint_path = "./checkpoints/resnet/"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 metrics=['val_accuracy'],
                                                 verbose=1)

# Loading Data and Configuring Model                                      

img_height = img_width = 224
batch_size = 25

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

# Transfer Learning w/ RESNET50V2 Model (using pretrained model as feature extractor)

preprocess_input = keras.applications.resnet_v2.preprocess_input

resnet_model = keras.applications.ResNet50V2(input_shape = [img_height,img_width,3], weights="imagenet", include_top=False)
resnet_model.trainable = False 
# resnet_model.summary()

inputs = keras.Input(shape=(img_height,img_width,3))
x = preprocess_input(inputs)
x = resnet_model(x, training = False)

# Custom Classification Head (Explore other possibilities)
x = layers.Flatten()(x)
outputs = layers.Dense(num_classes, activation = "softmax")(x)

model = keras.Model(inputs, outputs)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

num_epochs = 7
history = model.fit(
  ds_train,
  epochs=num_epochs,
  verbose=2,
  validation_data=ds_validation,
  callbacks=[cp_callback]
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
# plt.show()
plt.savefig("resnet_model.png", bbox_inches="tight")

# #Finding misclassified images
# file_paths = ds_validation.file_paths

# y_prediction = model.predict(ds_validation)
# y_prediction_class = np.argmax(y_prediction, axis=1)
# y_true = np.concatenate([y for x, y in ds_validation], axis=0)
# errors = np.where(y_prediction_class != y_true)
# print(type(errors))
# print(len(errors))

# # TODO: check if class names and model stuff is matching, make visualizer
# # TODO: save best model weights in gcp using checkpoint