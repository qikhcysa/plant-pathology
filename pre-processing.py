import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import RandAugment
import albumentations as A
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class WeatherAugmentation:
    def __init__(self):
        self.transforms = [
            self.solar_illumination(),
            self.rain_effect(),
            self.shadow_effect(),
            self.fog_effect()
        ]
 
    def solar_illumination(self):
        return A.RandomSunFlare(
            flare_roi=(0.9, 0, 1, 0.5),  # upper right corner
            flare_roi_2=(0.0, 0.0, 1.0, 0.1),  # upper left corner
            flare_radius=300,
        )
 
    def rain_effect(self):
        return A.RandomRain(
            rain_drop_size=1.0,
            rain_type='drizzle',
            brightness_coefficient=0.6,
        )
 
    def shadow_effect(self):
        return A.RandomShadow(
            num_shadows_lower=1,
            num_shadows_upper=5,
            shadow_dimension=6,   
        )
 
    def fog_effect(self):
        return A.RandomFog(
            fog_coef=(0.25, 0.8),
            alpha_coef_range=(0.25, 0.8),
            p=0.3
        )
 
    def apply_transforms(self, image):
        # Randomly select which transforms to apply
        active_transforms = [
            transform for transform in self.transforms
            if random.random() < 0.6
        ]
        # Compose the selected transforms
        composed_transform = A.Compose(active_transforms, p=1.0)
        # Apply the composed transform to the image
        transformed = composed_transform(image=image)['image']
        return transformed
    
if __name__ == "__main__":

 
    # Load an example image
    image = Image.open('path_to_your_image.jpg').convert('RGB')
    image = np.array(image)
 
    # Create an instance of the WeatherAugmentation class
    weather_aug = WeatherAugmentation()
 
    # Apply the augmentations
    augmented_image = weather_aug.apply_transforms(image)
 
    # Display the original and augmented images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
 
    plt.subplot(1, 2, 2)
    plt.title('Augmented Image')
    plt.imshow(augmented_image)
    plt.axis('off')
 
    plt.show()