import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2

class ResNet18Descriptor:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Quitar capa final
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def describe(self, image):
        if len(image.shape) == 2:  # gris -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        input_tensor = self.transform(image).unsqueeze(0)  # [1, 3, 224, 224]
        with torch.no_grad():
            feature = self.model(input_tensor).squeeze().numpy()

        feature = feature.flatten()
        feature /= (np.linalg.norm(feature) + 1e-8)  # Normalización L2
        return feature

