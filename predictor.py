
import torch
import torchvision.transforms as transforms
import cv2  # OpenCV for image loading
from torch import nn
from collections import OrderedDict

import numpy as np


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=5)
        )

        # Fixed in_features to match the checkpoint
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=256, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=1)
        )

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_model(x)
        x = torch.sigmoid(x)  # Binary classification output
        return x


# Load the model
model = CNN()

# # Load the model weights
model_path = "model/brain_tumor_detector.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# model.eval()


# Function to check the prediction
def check(input_img, model, device):
    print("Your image is: " + input_img)

    # Load the image using OpenCV
    img = cv2.imread(input_img)

    if img is None:
        print("Error: Image not found.")
        return None

    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Normalize the image
    img = img.transpose((2, 0, 1))  # Change to (C, H, W)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img_tensor = torch.from_numpy(img).float().to(device)

    # Make a prediction

    with torch.no_grad():
        output = model(img_tensor)
        predicted_value = output.item()  # Get the predicted score
    
    
    predicted = (output > 0.5).float()

    # Interpret prediction
    if predicted.item() == 1:
        status = True
    else:
        status = False

    
    print(f"Raw Output Value: {predicted_value}")
    print("Prediction status: ", status)
    
    return status, predicted_value



   
