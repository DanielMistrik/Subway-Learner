import torch
import torch.nn as nn
import torch.nn.functional as F

# Using
class SubwayCNN(nn.Module):
    def __init__(self):
        super(SubwayCNN, self).__init__()

        # Convolutional Features
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Dense Neural Network
        self.classifier = nn.Sequential(
            nn.Linear(16 * 10 * 11, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16 * 10 * 11)  # Flatten
        x = self.classifier(x)

        return x