import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionH4ptNet1(nn.Module):
    """
    PyTorch implementation of regression_H4pt_Net1
    
    Network architecture:
    - Conv block 1: 2x Conv2d (64 channels) + MaxPool
    - Conv block 2: 2x Conv2d (64 channels) + MaxPool  
    - Conv block 3: 2x Conv2d (128 channels)
    - FC layers: 3x Conv2d layers acting as fully connected
    """
    
    def __init__(self, input_channels=None):
        super(RegressionH4ptNet1, self).__init__()
        
        # Conv block 1
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 2
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Conv block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Fully connected layers (implemented as 1x1 and 4x4 convolutions)
        self.fc1 = nn.Conv2d(128, 128, kernel_size=4, padding=0)  # VALID padding
        self.fc2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.fc3 = nn.Conv2d(128, 8, kernel_size=1, padding=0)
        
    def forward(self, correlation):
        """
        Forward pass
        
        Args:
            correlation: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            H1_motion: Output tensor of shape (batch_size, 8, 1)
        """
        # Conv block 1
        x = F.relu(self.conv1_1(correlation))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool1(x)
        
        # Conv block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        
        # Conv block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for final layer
        
        # Reshape: squeeze spatial dimensions and add dimension at index 2
        # tf.squeeze(tf.squeeze(fc3,1),1) removes height and width dimensions
        # tf.expand_dims(..., [2]) adds dimension at index 2
        x = x.squeeze(-1).squeeze(-1)  # Remove spatial dimensions (H, W)
        H1_motion = x.unsqueeze(2)  # Add dimension at index 2: (batch, 8, 1)
        
        return H1_motion


class RegressionH4ptNet2(nn.Module):
    """
    PyTorch implementation of regression_H4pt_Net1
    
    Network architecture:
    - Conv block 1: 2x Conv2d (64 channels) + MaxPool
    - Conv block 2: 2x Conv2d (64 channels) + MaxPool  
    - Conv block 3: 2x Conv2d (128 channels)
    - FC layers: 3x Conv2d layers acting as fully connected
    """
    
    def __init__(self, input_channels=None):
        super(RegressionH4ptNet2, self).__init__()
        
        # Conv block 1
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 2
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Fully connected layers (implemented as 1x1 and 4x4 convolutions)
        self.fc1 = nn.Conv2d(128, 128, kernel_size=4, padding=0)  # VALID padding
        self.fc2 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.fc3 = nn.Conv2d(128, 8, kernel_size=1, padding=0)
        
    def forward(self, correlation):
        """
        Forward pass
        
        Args:
            correlation: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            H1_motion: Output tensor of shape (batch_size, 8, 1)
        """
        # Conv block 1
        x = F.relu(self.conv1_1(correlation))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool1(x)
        
        # Conv block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)
        
        # Conv block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for final layer
        
        # Reshape: squeeze spatial dimensions and add dimension at index 2
        # tf.squeeze(tf.squeeze(fc3,1),1) removes height and width dimensions
        # tf.expand_dims(..., [2]) adds dimension at index 2
        x = x.squeeze(-1).squeeze(-1)  # Remove spatial dimensions (H, W)
        H2_motion = x.unsqueeze(2)  # Add dimension at index 2: (batch, 8, 1)
        
        return H2_motion
    
class RegressionH4ptNet3(nn.Module):
    """
    PyTorch implementation of regression_H4pt_Net1
    
    Network architecture:
    - Conv block 1: 2x Conv2d (64 channels) + MaxPool
    - Conv block 2: 2x Conv2d (64 channels) + MaxPool  
    - Conv block 3: 2x Conv2d (128 channels)
    - FC layers: 3x Conv2d layers acting as fully connected
    """
    
    def __init__(self, input_channels=None,
                 grid_w=8,
                 grid_h=8
                 ):
        super(RegressionH4ptNet3, self).__init__()
        self.grid_w = grid_w
        self.grid_h = grid_h
        
        # Conv block 1
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 2
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        
        # Fully connected layers (implemented as 1x1 and 4x4 convolutions)
        self.fc1 = nn.Conv2d(256, 2048, kernel_size=4, padding=0)  # VALID padding
        self.fc2 = nn.Conv2d(2048, 1024, kernel_size=1, padding=0)
        self.fc3 = nn.Conv2d(1024, (grid_w+1)*(grid_h+1)*2, kernel_size=1, padding=0)
        
    def forward(self, correlation):
        """
        Forward pass
        
        Args:
            correlation: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            H1_motion: Output tensor of shape (batch_size, 8, 1)
        """
        # Conv block 1
        x = F.relu(self.conv1_1(correlation))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool1(x)
        
        # Conv block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool2(x)
        
        # Conv block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.maxpool3(x)
        
        # Conv block 4
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for final layer
        
        # Reshape: squeeze spatial dimensions and add dimension at index 2
        # tf.squeeze(tf.squeeze(fc3,1),1) removes height and width dimensions
        # tf.expand_dims(..., [2]) adds dimension at index 2
        x = x.squeeze(-1).squeeze(-1)  # Remove spatial dimensions (H, W)
        H3_motion = x.unsqueeze(2)  # Add dimension at index 2: (batch, 8, 1)
        
        return H3_motion