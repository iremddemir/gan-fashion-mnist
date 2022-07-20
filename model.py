import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
torch.manual_seed(42)
#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3,3), 2)
        self.conv2 = nn.Conv2d(16, 32, (5,5), 2)
        self.conv3 = nn.Conv2d(32, 64, (5,5), 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64,out_features = 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

#Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(64, 256, (3,3), 2)
        self.deconv2 = nn.ConvTranspose2d(256, 128, (4,4), 1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, (3,3), 2)
        self.final = nn.ConvTranspose2d(64, 1, (4,4), 2)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = x.view(-1, 64, 1, 1)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.final(x)
        x = self.tanh(x)
        return x

#helper functions
def discriminator_loss(pred):
    loss = nn.BCEWithLogitsLoss()
    gt = torch.zeros_like(pred)
    total_loss = loss(pred, gt)
    return total_loss

def generator_loss(pred):
    loss = nn.BCEWithLogitsLoss()
    gt = torch.ones_like(pred)
    total_loss = loss(pred, gt)
    return total_loss

def discriminator_optimizer(params, lr=0.001):
    return optim.Adam(params, lr=lr, betas=(0.5, 0.99))

def generator_optimizer(params, lr=0.001):
    return optim.Adam(params, lr=lr,betas=(0.5, 0.99))

def weights_init(m):  
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)