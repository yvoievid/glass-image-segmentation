import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
from PIL import Image

class JointTransform:
    def __init__(self):
        self.resize = transforms.Resize((256, 256))
        self.random_horizontal_flip = transforms.RandomHorizontalFlip()
        self.random_vertical_flip = transforms.RandomVerticalFlip()
        self.random_rotation = transforms.RandomRotation(degrees=30)

        self.transforms_image = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, image, mask):
        if not isinstance(image, Image.Image):
            image = F.to_pil_image(image)
        if not isinstance(mask, Image.Image):
            mask = F.to_pil_image(mask)

        image = self.resize(image)
        mask = self.resize(mask)

        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)

        angle = random.uniform(-30, 30)
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle)

        image = self.transforms_image(image)
        mask = F.to_tensor(mask)

        return image, mask
