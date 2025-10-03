#!/usr/bin/env python3
"""
Test script to create a simple test image and run string art with perceptual loss
"""

import numpy as np
from PIL import Image, ImageDraw
import os

def create_test_image():
    """Create a simple test image with some geometric shapes"""
    # Create a white canvas
    img = Image.new('L', (256, 256), 255)
    draw = ImageDraw.Draw(img)
    
    # Draw a black circle in the center
    center = 128
    radius = 60
    draw.ellipse([center-radius, center-radius, center+radius, center+radius], fill=0)
    
    # Draw some additional shapes for more interesting features
    draw.rectangle([50, 50, 100, 100], fill=100)  # Gray square
    draw.ellipse([150, 50, 200, 100], fill=50)    # Dark ellipse
    
    return img

def main():
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and save test image
    test_img = create_test_image()
    test_img_path = os.path.join(output_dir, "test_image.png")
    test_img.save(test_img_path)
    
    print(f"Created test image: {test_img_path}")
    print("You can now run the main script and select this test image to test the perceptual loss functionality!")
    
    # Also show some info about using the script
    print("\nTo test the perceptual loss string art:")
    print("1. Run: python plossapp.py")
    print("2. Select the test_image.png when prompted")
    print("3. The script will use perceptual loss to generate string art")

if __name__ == "__main__":
    main()
