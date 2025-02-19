import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First Conv Layer
        x = self.conv1(x)  # 28x28 -> 28x28
        x = F.relu(x)
        x = self.pool1(x)  # 28x28 -> 14x14

        # Second Conv Layer
        x = self.conv2(x)  # 14x14 -> 14x14
        x = F.relu(x)
        x = self.pool2(x)  # 14x14 -> 7x7

        # Flatten
        x = x.view(-1, 64 * 7 * 7)

        # Fully Connected
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)
