import torch
import torch.nn as nn
from torchvision import models

class SimpleEncoder(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=128):
        super(SimpleEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x16x16
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x8x8
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 128x4x4

        self.flatten = nn.Flatten()
        self.fc_embedding = nn.Linear(128 * 4 * 4, embedding_dim)
        self.relu_embedding = nn.ReLU()
        self.fc_classifier = nn.Linear(embedding_dim, num_classes)

    def get_embeddings(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu_embedding(self.fc_embedding(x))
        return x

    def forward(self, x):
        embeddings = self.get_embeddings(x)
        output = self.fc_classifier(embeddings)
        return output

class BackdoorModel(nn.Module):
    """
    Model for backdoor attacks and detection
    
    Has separate paths for feature extraction and classification
    """
    def __init__(self, num_classes=10, feature_dim=128):
        super(BackdoorModel, self).__init__()
        
        # Base backbone (ResNet18)
        self.backbone = models.resnet18(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove classifier
        
        # Feature embedding layer
        self.feature_layer = nn.Sequential(
            nn.Linear(num_ftrs, feature_dim),
            nn.ReLU()
        )
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def get_embeddings(self, x):
        """Extract features without classification"""
        features = self.backbone(x)
        embeddings = self.feature_layer(features)
        return embeddings
    
    def forward(self, x, return_features=False):
        """Full forward pass with optional feature return"""
        features = self.backbone(x)
        embeddings = self.feature_layer(features)
        
        if return_features:
            return embeddings
            
        logits = self.classifier(embeddings)
        return logits