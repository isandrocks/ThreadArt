#!/usr/bin/env python3
"""
Simple test script to verify that the perceptual loss functionality works
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

class PerceptualLoss(nn.Module):
    """Perceptual loss using pretrained VGG16 features"""
    def __init__(self, device='cpu'):
        super(PerceptualLoss, self).__init__()
        self.device = device
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.feature_layers = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[:9],   # relu2_2
            vgg[:16],  # relu3_3
            vgg[:23],  # relu4_3
        ])
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.to(device)
        self.eval()
          # Normalization for ImageNet pretrained models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
      def forward(self, x, y):
        """
        Compute perceptual loss between x and y
        x, y: tensors of shape (B, C, H, W) in range [0, 1]
        """
        # Convert grayscale to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        
        # Normalize - but only after converting to RGB
        x = self.normalize(x)
        y = self.normalize(y)
        
        loss = 0.0
        
        # Extract features from multiple layers
        for layer in self.feature_layers:
            x_features = layer(x)
            y_features = layer(y)
            
            # Compute MSE loss between features
            loss += F.mse_loss(x_features, y_features)
        
        return loss

def test_perceptual_loss():
    """Test the perceptual loss functionality"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize perceptual loss network
    perceptual_loss = PerceptualLoss(device=device)
    
    # Create two test images
    # Image 1: All black
    img1 = np.zeros((64, 64), dtype=np.uint8)
    # Image 2: All white  
    img2 = np.ones((64, 64), dtype=np.uint8) * 255
    # Image 3: Similar to img1 (slightly gray)
    img3 = np.ones((64, 64), dtype=np.uint8) * 50
    
    # Convert to tensors
    tensor1 = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor2 = torch.from_numpy(img2).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor3 = torch.from_numpy(img3).float().unsqueeze(0).unsqueeze(0) / 255.0
    
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device) 
    tensor3 = tensor3.to(device)
    
    # Test perceptual loss calculations
    with torch.no_grad():
        # Loss between very different images (black vs white)
        loss_1_2 = perceptual_loss(tensor1, tensor2).item()
        # Loss between similar images (black vs dark gray)
        loss_1_3 = perceptual_loss(tensor1, tensor3).item()
        # Loss between same images (should be 0)
        loss_1_1 = perceptual_loss(tensor1, tensor1).item()
    
    print(f"Loss between black and white images: {loss_1_2:.6f}")
    print(f"Loss between black and gray images: {loss_1_3:.6f}")
    print(f"Loss between identical images: {loss_1_1:.6f}")
    
    # Verify that the losses make sense
    assert loss_1_2 > loss_1_3, "Loss between very different images should be higher"
    assert abs(loss_1_1) < 1e-6, "Loss between identical images should be near zero"
    
    print("✓ Perceptual loss test passed!")
    
    return True

if __name__ == "__main__":
    try:
        test_perceptual_loss()
        print("All tests passed! The perceptual loss functionality is working correctly.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
