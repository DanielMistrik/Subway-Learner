import torch
import torch.nn as nn
import torch.nn.functional as F

# Using
class SubwayCNN(nn.Module):
    def __init__(self):
        super(SubwayCNN, self).__init__()

        # Convolutional Features
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Dense Neural Network
        self.classifier = nn.Sequential(
            nn.Linear(16 * 20 * 22, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 20 * 22)  # Flatten
        x = self.classifier(x)

        return x