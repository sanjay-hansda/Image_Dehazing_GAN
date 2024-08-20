TEST_DATASET_HAZY_PATH = '/home/user/Desktop/test/hazy'
TEST_DATASET_OUTPUT_PATH = '/home/user/Desktop/test/dehazed_output'

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input
import matplotlib.image as mpimg

input_images = os.listdir(TEST_DATASET_HAZY_PATH)
num = len(input_images)
output_images = []

for i in range(num):
    output_images.append(os.path.join(TEST_DATASET_OUTPUT_PATH, input_images[i]))
    input_images[i] = os.path.join(TEST_DATASET_HAZY_PATH, input_images[i])

# Load the saved model weights
gan = load_model('pix2pix.h5')
gan.img_rows = 256
gan.img_cols = 256
gan.channels = 3
gan.img_shape = (gan.img_rows, gan.img_cols, gan.channels)
gan.generator = gan.get_layer('model_13')  # Replace 'model_13' with the model with higher number if error report occurs
img_A = Input(shape=gan.img_shape)
img_B = Input(shape=gan.img_shape)
fake_A = gan.generator(img_B)
image_size=(256, 256)


for i in range(10):
    img = tf.io.read_file(input_images[i])
    img = tf.image.decode_image(img, channels=3)  # Ensure RGB channels
    img = tf.image.resize(img, image_size)
    img = img / 255.0  # Normalize pixel values
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    dehazed_image = gan.generator.predict(img)
    dehazed_image = np.clip(dehazed_image, 0, 1)
    filename = os.path.basename(input_images[i])
    output_file = os.path.join(TEST_DATASET_OUTPUT_PATH, filename)
    mpimg.imsave(output_file, dehazed_image[0])