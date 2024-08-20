import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image

# Specify the root directory
root_directory = 'final_dataset (train+val)/train'

# Custom function to adjust brightness and contrast
def adjust_brightness_contrast(image, brightness_factor, contrast_factor):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Adjust brightness
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_factor, 0, 255)

    # Convert back to RGB color space
    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Adjust contrast
    mean_intensity = np.mean(rgb_image)
    rgb_image = (rgb_image - mean_intensity) * contrast_factor + mean_intensity
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    return rgb_image

class ImageAugmentor:
    def __init__(self, root_dir, num_images_to_augment=100):
        self.root_dir = root_dir
        self.num_images_to_augment = num_images_to_augment

        # Define the directories for hazy and GT images
        self.hazy_dir = os.path.join(root_dir, 'hazy')
        self.gt_dir = os.path.join(root_dir, 'GT')

        # Create directories to save augmented images
        self.augmented_hazy_dir = os.path.join(root_dir, 'augmented', 'hazy')
        self.augmented_gt_dir = os.path.join(root_dir, 'augmented', 'GT')
        os.makedirs(self.augmented_hazy_dir, exist_ok=True)
        os.makedirs(self.augmented_gt_dir, exist_ok=True)

        # Initialize image data generator for augmentation
        self.data_generator = ImageDataGenerator(
            horizontal_flip=False,  # Random horizontal flipping
            vertical_flip=False,    # Random vertical flipping
        )

    def augment_images(self):
        hazy_files = os.listdir(self.hazy_dir)
        num_augmented = 0

        for filename in hazy_files:
            if num_augmented >= self.num_images_to_augment:
                break

            # Load hazy image
            hazy_path = os.path.join(self.hazy_dir, filename)
            hazy_img = np.array(Image.open(hazy_path))

            # Load corresponding GT image
            gt_path = os.path.join(self.gt_dir, filename)
            gt_img = np.array(Image.open(gt_path))

            # Reshape images to comply with Keras data format (height, width, channels)
            hazy_arr = hazy_img.reshape((1,) + hazy_img.shape)
            gt_arr = gt_img.reshape((1,) + gt_img.shape)

            # Generate augmented images
            hazy_augmented = self.data_generator.flow(hazy_arr, batch_size=1)
            gt_augmented = self.data_generator.flow(gt_arr, batch_size=1)

            # Retrieve augmented images
            hazy_augmented = next(hazy_augmented)[0]
            gt_augmented = next(gt_augmented)[0]

            # Adjust brightness and contrast of hazy augmented image
            brightness_factor = np.random.uniform(0.5, 1.5)
            contrast_factor = np.random.uniform(0.5, 1.5)
            hazy_augmented = adjust_brightness_contrast(hazy_augmented, brightness_factor, contrast_factor)
            gt_augmented = adjust_brightness_contrast(gt_augmented, brightness_factor, contrast_factor)

            # Save augmented images
            hazy_augmented_img = Image.fromarray(hazy_augmented.astype('uint8'))
            gt_augmented_img = Image.fromarray(gt_augmented.astype('uint8'))

            hazy_augmented_img.save(os.path.join(self.augmented_hazy_dir, f"{filename.split('.')[0]}_augmented.jpg"))
            gt_augmented_img.save(os.path.join(self.augmented_gt_dir, f"{filename.split('.')[0]}_GT_augmented.jpg"))

            num_augmented += 1
            if(num_augmented %25==0):
              print(num_augmented)
        print(f"{num_augmented} images augmented and saved in {self.augmented_hazy_dir} and {self.augmented_gt_dir}")


# Create ImageAugmentor object and augment images
augmentor = ImageAugmentor(root_directory, num_images_to_augment=7619)
augmentor.augment_images()