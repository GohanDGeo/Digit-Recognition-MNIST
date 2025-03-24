import streamlit as st
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps

# Parent class for the network, so both a network with linear layers and a CNN network can be tested.
class DigitClassifier(nn.Module):
    
    def __init__(self) -> None:
        super(DigitClassifier, self).__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward method") 

    # A function tha trains the network based on the given train data, optimizer and criterion.
    # args:
    # @train_loader -> The dataloader for the train data
    # @optimizer ->  The optimizer to be used during training
    # @criterion -> The criterion to be used for training
    # @num_epochs -> The number of epochs to train for
    # @verbose -> Set to True if training updates are to be printed out
    def train_model(self, train_loader, optimizer, criterion, num_epochs=10, verbose=True):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()  # Reset gradients
                outputs = self(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights

                running_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
        if verbose:
            print("Training Complete!")
        return running_loss/len(train_loader)

    # A function that runs the model on test data.
    # args:
    # @test_loader -> The dataloader for the test data
    # @verbose -> Set to True if test accuracy is to be printed out
    def test_model(self, test_loader, verbose=True):
        correct = 0
        total = 0

        self.eval()

        with torch.no_grad():  # No need to track gradients
            for images, labels in test_loader:
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)  # Get class with highest probability
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        
        if verbose:
            print(f"Test Accuracy: {accuracy:.2f}%")

        return accuracy

# A subclass of DigitClassifier that uses Convolutional Layers and maxpooling
class CNNDigitClassifier(DigitClassifier):
    
    def __init__(self) -> None:
        super(CNNDigitClassifier, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, -1)
    
file_path = os.path.join("network-files", "mnist_model.pth")

# Load the trained model
model = CNNDigitClassifier()
model.load_state_dict(torch.load(file_path, map_location=torch.device("cpu")))  
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
    img.save("image1.png")
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
        _, predicted = torch.max(probs, dim=1)
    
    # Show Prediction
    st.write(f"**Predicted Digit: {predicted.item()}**")
    st.write(probs)
    st.bar_chart(probs.numpy().flatten())  # Display probabilities for each digit
