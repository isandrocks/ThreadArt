from PIL import Image, ImageOps
import numpy as np
import math
import collections
from skimage.transform import radon
import svgwrite
from SAapp import new_func
import os

import tkinter as tk
from tkinter import filedialog

# Create a Tkinter root window (it will not be shown)
root = tk.Tk()
root.withdraw()

# Open a file dialog to select the file
file_path = filedialog.askopenfilename(
    title="Select an image file",
    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
)

SET_LINES = 0

N_PINS = 36 * 8
if SET_LINES != 0:
    MAX_LINES = SET_LINES
else:
    MAX_LINES = int(((N_PINS * (N_PINS - 1)) // 2))
MIN_LOOP = 5
MIN_DISTANCE = 35
LINE_WEIGHT = 8
FILENAME = file_path
SCALE = 18
img = Image.open(FILENAME)

# Get the dimensions of the image
width, height = img.size

# Calculate the new dimensions while maintaining aspect ratio
if width > 512 or height > 512:
  if width < height:
    new_width = 512
    new_height = int(height * (512 / width))
  else:
    new_width = int(width * (512 / height))
    new_height = 512
else:
  new_width = width
  new_height = height

resized_image = img.resize((new_width, new_height))

if resized_image.size[0] != resized_image.size[1]:
  new_image = resized_image.crop((new_width // 2 - 256, new_height // 2 - 256, new_width // 2 + 256, new_height // 2 + 256))
else:
  new_image = resized_image


img = new_image
if img.mode == "RGBA":
  img = img.convert("RGB")
img = ImageOps.invert(img)

def string_art_cmyk(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img):
    # Convert image to CMYK
    img_cmyk = img.convert("CMYK")
  
    cmyk_channels = [np.array(img_cmyk.getchannel(channel)) for channel in range(4)]
    
    results = []  # To store results for each channel
    lengths = []  # To store line counts for each channel
    diffs = []    # To store errors for each channel

    # Process each channel
    for channel_idx, channel_img in enumerate(cmyk_channels):
        print(f"Processing channel {['Cyan', 'Magenta', 'Yellow', 'Black'][channel_idx]}...")
        length, result, line_number, current_absdiff = new_func(
            N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, channel_img
        )
        results.append(result)
        lengths.append(line_number)
        diffs.append(current_absdiff)

    # Combine the results into a single CMYK image
    combined_result = Image.merge(
        "CMYK", [Image.fromarray(np.array(res)) for res in results]
    )
    return combined_result, lengths, diffs

result, length, current_absdiff = string_art_cmyk(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img)
print(f"Total lines: {sum(length)}")
print('\x07')

result_1024 = result.resize((1024, 1024), Image.Resampling.LANCZOS)
result_1024 = result_1024.convert("RGB")
result_1024 = ImageOps.invert(result_1024)

result_1024.save(os.path.splitext(FILENAME)[0] + f"_LW_{LINE_WEIGHT}".replace('.', '_') + ".png")
