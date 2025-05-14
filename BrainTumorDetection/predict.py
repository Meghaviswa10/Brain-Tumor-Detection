import torch
from torchvision import transforms
from PIL import Image
from model import load_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = load_model(num_classes=2).to(device)
model.load_state_dict(torch.load('saved_models/trained_model.pth', map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        class_names = ['No Tumor', 'Tumor']
        prediction = class_names[predicted.item()]
    return prediction
