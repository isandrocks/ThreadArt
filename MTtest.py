import concurrent.futures
import cupy as cp
from PIL import Image, ImageEnhance, ImageOps
import tkinter as tk
from tkinter import filedialog
import numpy as np
from SAapp import new_func
import os

root = tk.Tk()
root.withdraw()

# Open a file dialog to select the file
file_path = filedialog.askopenfilename(
      title="Select an image file",
      filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
  )

def process_channel(channel_img, N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE):
    # Convert the image to a CuPy array for GPU processing
    channel_img = cp.array(channel_img)
    
    # Perform your image processing tasks here using CuPy
    # For example, you can use cp.mean() instead of np.mean()
    
    # Convert back to a NumPy array if needed
    channel_img = cp.asnumpy(channel_img)
    
    # Call your existing function
    length, result, line_number, current_absdiff = new_func(
        N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, channel_img
    )
    return result, line_number, current_absdiff

def string_art_cmyk(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img):
    img_cmyk = img.convert("CMYK")
    cmyk_channels = [np.array(img_cmyk.getchannel(channel)) for channel in range(4)]
    
    results = []
    lengths = []
    diffs = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_channel, channel_img, N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE) for channel_img in cmyk_channels]
        for future in concurrent.futures.as_completed(futures):
            result, line_number, current_absdiff = future.result()
            results.append(result)
            lengths.append(line_number)
            diffs.append(current_absdiff)

    combined_result = Image.merge(
        "CMYK", [Image.fromarray(np.array(res)) for res in results]
    )
    return combined_result, lengths, diffs

def main():
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog to select the file
    FILENAME = filedialog.askopenfilename()

    # Load your image
    img = Image.open(FILENAME)

    # Ensure the image is square
    size = min(img.size)
    img = img.resize((size, size), Image.Resampling.LANCZOS)

    SET_LINES = 0

    N_PINS = 36 * 8
    if SET_LINES != 0:
        MAX_LINES = SET_LINES
    else:
        MAX_LINES = int(((N_PINS * (N_PINS - 1)) // 2))
    MIN_LOOP = 5
    MIN_DISTANCE = 35
    LINE_WEIGHT = 2
    SCALE = 25

    result, length, current_absdiff = string_art_cmyk(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img)
    print(f"Total lines: {sum(length)}")
    print('\x07')

    result_1024 = result.resize((1024, 1024), Image.Resampling.LANCZOS)
    result_1024 = result_1024.convert("RGB")
    result_1024.save(os.path.splitext(FILENAME)[0] + f"_LW_{LINE_WEIGHT}".replace('.', '_') + ".png")

if __name__ == "__main__":
    main()