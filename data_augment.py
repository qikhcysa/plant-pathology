import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import RandAugment
import albumentations as A
import random
import numpy as np

def augmentfunc(img):
    img_shape = img.shape
    # 定义图像预处理和增强的转换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        RandAugment(N=2, M=9),  # N is the number of augmentation transformations to apply sequentially, M is the magnitude for all the transformations.
        transforms.Resize((img_shape[1], img_shape[2]), interpolation=InterpolationMode.BILINEAR),  # resize back to original size
        transforms.ToTensor(),
    ])
    # 应用数据增强
    augmented_image = transform(img)
    # 现在augmented_image是一个增强后的图像，它仍然是一个[3, 384, 384]的torch.Tensor
    return augmented_image


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
            p=0.6
        )
 
    def rain_effect(self):
        return A.RandomRain(
            rain_drop_size=1.0,
            rain_type='drizzle',
            brightness_coefficient=0.6,
            p=0.6
        )
 
    def shadow_effect(self):
        return A.RandomShadow(
            num_shadows_lower=1,
            num_shadows_upper=5,
            shadow_dimension=6,
            p=0.6
        )
 
    def fog_effect(self):
        return A.RandomFog(
            fog_coef=(0.25, 0.8),
            alpha_coef_range=(0.25, 0.8),
            p=0.6
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