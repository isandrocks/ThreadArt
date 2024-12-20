from PIL import Image
import numpy as np

def convert_to_grayscale(image_path, output_path):
  # Open the image file
  img = Image.open(image_path).convert('L')
  
  # Convert image to numpy array
  img_array = np.asarray(img)
  
  # Normalize pixel values to be between 0 and 1
  normalized_array = img_array / 255.0
  
  # Save the normalized array to a file
  np.savetxt(output_path, normalized_array, fmt='%.4f')

if __name__ == "__main__":
  input_image_path = '1351093813.png'  # Replace with your image file path
  output_file_path = 'output_pixels.txt'  # Replace with your desired output file path
  
  convert_to_grayscale(input_image_path, output_file_path)