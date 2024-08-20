from __future__ import print_function, division
import os
import random
import warnings
import numpy as np
import datetime
import tensorflow as tf
from keras.optimizers.legacy import Adam as LegacyAdam
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, LeakyReLU, UpSampling2D, Dropout, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Specify the root directory
root_directory = 'final_dataset (train+val)'


class ImageDataset:
    def __init__(self, root_dir, mode='train', batch_size=32, image_size=(256, 256)):
        self.root_dir = os.path.join(root_dir, mode)
        self.mode = mode
        self.batch_size = batch_size
        self.image_size = image_size

        # Define the directories for hazy and GT images based on the mode
        self.hazy_dir = os.path.join(self.root_dir, 'hazy')
        self.gt_dir = os.path.join(self.root_dir, 'GT')

        # Get the list of image filenames
        self.hazy_files = sorted(os.listdir(self.hazy_dir))
        self.gt_files = sorted(os.listdir(self.gt_dir))

        # Calculate the number of batches
        self.num_batches = len(self.hazy_files) // batch_size

    def _read_image(self, file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3)  # Ensure RGB channels
        img = tf.image.resize(img, self.image_size)
        img = img / 255.0  # Normalize pixel values
        return img

    def __iter__(self):
        return self

    def __next__(self):
        # Check if there are still batches left to iterate through
        if self.num_batches == 0:
            raise StopIteration

        # Initialize empty lists to store hazy and GT images
        batch_hazy = []
        batch_gt = []

        # Iterate through the current batch of filenames
        start_idx = self.num_batches * self.batch_size
        end_idx = min((self.num_batches + 1) * self.batch_size, len(self.hazy_files))
        for i in range(start_idx, end_idx):
            # Load hazy image
            hazy_path = os.path.join(self.hazy_dir, self.hazy_files[i])
            hazy_img = self._read_image(hazy_path)
            batch_hazy.append(hazy_img)

            # Load GT image
            gt_path = os.path.join(self.gt_dir, self.gt_files[i])
            gt_img = self._read_image(gt_path)
            batch_gt.append(gt_img)

        # # Convert lists to TensorFlow tensors
        batch_hazy = tf.convert_to_tensor(batch_hazy)
        batch_gt = tf.convert_to_tensor(batch_gt)

        # Decrement the number of batches
        self.num_batches -= 1

        return batch_hazy, batch_gt

    def __len__(self):
        return self.num_batches




# Create validation dataset using the optimized data generator
train_dataset = ImageDataset(root_dir=root_directory, mode='train', batch_size=32)


# Suppress deprecation warnings related to LeakyReLU activation
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

class Pix2Pix():
    def __init__(self, data_loader):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.dataset_name = "train-dataset"

        # Configure data loader
        self.data_loader = data_loader

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = LegacyAdam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)


    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        # Sigmoid activation to ensure output in range [0, 1]
        validity = Activation('sigmoid')(validity)

        discriminator = Model([img_A, img_B], validity)
        return discriminator


    def train(self, epochs, batch_size=32, sample_interval=50):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        total_batches = 236

        for epoch in range(epochs):
            for batch_i, (batch_hazy, batch_gt) in enumerate(self.data_loader):
                batch_gt_np = batch_gt.numpy()
                batch_hazy_np = batch_hazy.numpy()
                # Ensure imgs_A and imgs_B have the same number of samples
                imgs_A = batch_gt_np
                imgs_B = batch_hazy_np

                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)

                if (epoch % 2)==0:
                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid[:len(batch_hazy)])
                    d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake[:len(batch_hazy)])
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Train Generator
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid[:len(batch_hazy)], imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                if(epoch % 2)==0:
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, total_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))
                else:
                    print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, total_batches,
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, batch_hazy_np, batch_gt_np)

                if batch_i % 75 == 74:
                    model_name = 'gan_32.h5'
                    model_path = os.path.join(root_directory, model_name)
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    self.combined.save(model_path, save_format='tf')


    def sample_images(self, epoch, batch_i, batch_hazy_np, batch_gt_np, num_samples=3):

        r, c = 3, num_samples
        fake_A = self.generator.predict(batch_hazy_np)

        fig, axs = plt.subplots(r, c)
        for i in range(r):

            # Get a random sample from the data
            hazy = batch_hazy_np[i]
            fake = fake_A[i]
            gt = batch_gt_np[i]

            # Generate the corresponding fake image
            hazy_normalized = hazy.astype(np.float32) * 255.0
            fake_normalized = fake.astype(np.float32) * 255.0
            gt_normalized = gt.astype(np.float32) * 255.0

            # Plot the images
            axs[i, 0].imshow(hazy)
            axs[i, 0].set_title('Condition')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(fake)
            axs[i, 1].set_title('Generated')
            axs[i, 1].axis('off')

            axs[i, 2].imshow(gt)
            axs[i, 2].set_title('Original')
            axs[i, 2].axis('off')

        save_path = os.path.join(root_directory, "{}_{}.png".format(epoch, batch_i))
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    gan = Pix2Pix(train_dataset)
    gan.train(epochs=10, batch_size=32, sample_interval=20)