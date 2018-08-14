import torch
from torchvision import datasets, models, transforms
from PIL import Image

model = torch.load('api/models/model_v1.pkl')
model.eval()


def image_loader(loader, file):
    image = Image.open(file)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
