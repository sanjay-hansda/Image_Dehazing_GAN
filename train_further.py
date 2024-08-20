from __future__ import print_function, division
import os
import random
import warnings
import numpy as np
import datetime
import tensorflow as tf
from keras.optimizers.legacy import Adam as LegacyAdam
from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, LeakyReLU, UpSampling2D, Dropout, Activation
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


# Specify the root directory, model path and directory to save new model
root_directory = '/content/drive/MyDrive/DL_Project2/final_dataset (train+val)'
model_path = '/content/drive/MyDrive/DL_Project2/final_dataset (train+val)/gan_32.h5'
new_model_path = '/content/drive/MyDrive/DL_Project2/final_dataset (train+val)'


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
            self.num_batches = len(self.hazy_files) // batch_size
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

if __name__ == '__main__':
    # Load the saved model weights
    gan = load_model(model_path)
    gan.img_rows = 256
    gan.img_cols = 256
    gan.channels = 3
    gan.img_shape = (gan.img_rows, gan.img_cols, gan.channels)

    # Configure data loader
    gan.data_loader = train_dataset

    # Extract generator and discriminator models from the loaded model
    gan.generator = gan.get_layer('model_16')  # Replace 'model_16' with the model with higher number if error report occurs
    gan.discriminator = gan.get_layer('model_15') # Replace 'model_15' with the model with lower number if error report occurs

    # Calculate output shape of D (PatchGAN)
    patch = int(gan.img_rows / 2**4)
    gan.disc_patch = (patch, patch, 1)

    # Number of filters in the first layer of G and D
    gan.gf = 64
    gan.df = 64

    optimizer = LegacyAdam(0.0002, 0.5)

    # Input images and their conditioning images
    img_A = Input(shape=gan.img_shape)
    img_B = Input(shape=gan.img_shape)

    # By conditioning on B generate a fake version of A
    fake_A = gan.generator(img_B)

    # For the combined model we will only train the generator
    gan.discriminator.trainable = False

    # Discriminators determines validity of translated images / condition pairs
    valid = gan.discriminator([fake_A, img_B])
    # Build and compile the discriminator
    gan.discriminator.compile(loss='mse',
        optimizer=optimizer,
        metrics=['accuracy'])

    gan.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
    gan.combined.compile(loss=['mse', 'mae'],
                            loss_weights=[1, 100],
                            optimizer=optimizer)
    
    """TRAIN CODE"""
    
    start_epoch = 1
    epochs = 5
    batch_size=32
    sample_interval=20


    start_time = datetime.datetime.now()

    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + gan.disc_patch)
    fake = np.zeros((batch_size,) + gan.disc_patch)

    total_batches = 236

    for epoch in range(start_epoch, epochs+start_epoch):
        for batch_i, (batch_hazy, batch_gt) in enumerate(gan.data_loader):
            batch_gt_np = batch_gt.numpy()
            batch_hazy_np = batch_hazy.numpy()
            # Ensure imgs_A and imgs_B have the same number of samples
            imgs_A = batch_gt_np
            imgs_B = batch_hazy_np

            # Condition on B and generate a translated version
            fake_A = gan.generator.predict(imgs_B)

            if (epoch % 2)==0:
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = gan.discriminator.train_on_batch([imgs_A, imgs_B], valid[:len(batch_hazy)])
                d_loss_fake = gan.discriminator.train_on_batch([fake_A, imgs_B], fake[:len(batch_hazy)])
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            g_loss = gan.combined.train_on_batch([imgs_A, imgs_B], [valid[:len(batch_hazy)], imgs_A])

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
                num_samples = 3
                r, c = 3, num_samples
                fake_A = gan.generator.predict(batch_hazy_np)

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

            if batch_i % 75 == 74:
                model_temp_name = 'retrained_gan_32.h5'
                model_path = os.path.join(new_model_path, model_temp_name)
                if os.path.exists(model_path):
                    os.remove(model_path)
                gan.combined.save(model_path, save_format='tf')
