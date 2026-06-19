import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def MNIST_MLP():
	model = nn.Sequential(
		nn.Flatten(),
		nn.Linear(784, 100),
		nn.ReLU(),
		nn.Linear(100, 100),
		nn.ReLU(),
		nn.Linear(100, 10)
	)
	return model

class MNIST_ConvSmall(nn.Module):
    def __init__(self):
        super(MNIST_ConvSmall, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*7*7, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x_p = torch.relu(self.fc1(x))
        x = self.fc2(x_p)
        return x

class MNIST_ConvLarge(nn.Module):
    def __init__(self):
        super(MNIST_ConvLarge, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def CIFAR10_CNN_A():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*8*8,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def CIFAR10_CNN_B():
    return nn.Sequential(
        nn.ZeroPad2d((1,2,1,2)),
        nn.Conv2d(3, 32, (5,5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4,4), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8192, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )

def CIFAR10_CNN_C():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(576, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model

def CIFAR10_ConvSmall():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32*6*6,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

def CIFAR10_ConvDeep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8*8*8, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )
    return model

class CIFAR10_ConvLarge(nn.Module):
    def __init__(self):
        super(CIFAR10_ConvLarge, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MicroPilotNet(nn.Module):
    def __init__(self):
        super(MicroPilotNet, self).__init__()
        # Input: 3 channels (RGB), 37 height, 117 width
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        # Flattened size: 64 channels * 2 height * 12 width = 1536
        self.linear_layers = nn.Sequential(
            nn.Linear(1536, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1) # Single regression output
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)

class CarlaSteeringNet(nn.Module):
    def __init__(self):
        super(CarlaSteeringNet, self).__init__()
        # Input: 3 channels (RGB), 60 height, 80 width (preserves 4:3 aspect ratio)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),  # 60x80 -> 30x40
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(), # 30x40 -> 15x20
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(), # 15x20 -> 8x10
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), nn.ReLU()  # 8x10 -> 4x5
        )
        # Flattened size: 64 channels * 4 height * 5 width = 1280
        self.linear_layers = nn.Sequential(
            nn.Linear(1280, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1) # Single regression output for steering angle
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.linear_layers(x)