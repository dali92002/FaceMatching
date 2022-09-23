import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    """
    A model to encode a batch of images of size [batch_size, channels, height, width] to batch
    of vectors with size [batch_size, 128]

    """
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1)
        

        self.dropout1 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(12800 , 128)
        self.fc2 = nn.Linear(256, 128)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        
        return x



class FaceEmb(nn.Module):
    """
    Implementation of the model that embedd a 3 images of faces and return their vectors
    and return the vectors.
    
    """
    def __init__(self):
        super(FaceEmb, self).__init__()
        self.backbone = Backbone()
        
    def forward(self, a , p, n):
        
        a = self.backbone(a)
        p = self.backbone(p)
        n = self.backbone(n)

        return a, p, n