import streamlit as st
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

class DigitClassifier(nn.Module):
    
    def __init__(self) -> None:
        super(DigitClassifier, self).__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(x, -1)
    
file_path = os.path.join("network-files", "mnist_full_model.pth")

# Load the trained model
model = torch.load(file_path, weights_only=False)
model.eval()  # Set to evaluation mode

# Streamlit App
st.title("🎨 MNIST Digit Recognizer")
st.write("Draw a digit below and let the model predict it!")

# Create a drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Transform function to process the drawn image
def preprocess_image(img):
    img = img.resize((28, 28), Image.ANTIALIAS)  # Resize to MNIST format
    img = img.convert("L")  # Convert to grayscale
    img = transforms.ToTensor()(img).unsqueeze(0)  # Convert to tensor and add batch dim
    return img

# Predict if the user has drawn something
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    img_tensor = preprocess_image(img)

    # Make prediction
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
    
    # Show Prediction
    st.write(f"**Predicted Digit: {prediction}**")
    st.bar_chart(probs.numpy().flatten())  # Display probabilities for each digit
