import torchvision.transforms as transforms

class BaseTransform:
    def __init__(self):
        self.transforms_image_and_mask = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.transforms_image = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, image, mask):
        image = self.transforms_image_and_mask(image)
        mask = self.transforms_image_and_mask(mask)
        image = self.transforms_image(image)
        return image, mask
