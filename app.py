
import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the class names in the same order as ImageFolder
import os

classes = sorted(os.listdir("train"))
classes = [
    name for name in classes
    if os.path.isdir(os.path.join("train", name))
]

# Same preprocessing as your test dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Recreate the trained ResNet18 architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 7)

model.load_state_dict(
    torch.load(
        "resnet.pth",
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()


def predict(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]

    return {
        classes[i]: float(probabilities[i])
        for i in range(len(classes))
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Currency Image"),
    outputs=gr.Label(num_top_classes=7, label="Prediction"),
    title="Currency Image Classifier",
    description="Upload a currency image to classify it using your trained ResNet18 model."
)

demo.launch()
