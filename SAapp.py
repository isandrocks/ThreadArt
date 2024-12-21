from skimage.transform import radon
import collections
import math
import os
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import numpy as np
import time
import svgwrite
import tkinter as tk
from tkinter import filedialog

def new_func(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img, LINE_COLOR):
    assert img.shape[0] == img.shape[1]
    length = img.shape[0]

# Apply circular mask
    X, Y = np.ogrid[0:length, 0:length]
    circlemask = (X - length / 2) ** 2 + (Y - length / 2) ** 2 > (length / 2) ** 2
    img[circlemask] = 0xFF

# Compute the Radon Transform of the image
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)

# Normalize the sinogram for line weighting
    sinogram /= sinogram.max()

# Precompute the scaling factor for distance
    distance_scale_factor = sinogram.shape[0] / (length / 2)

# Calculate pin coordinates
    pin_coords = []
    center = length / 2
    radius = length / 2 - 0.5

    for i in range(N_PINS):
        angle = 2 * math.pi * i / N_PINS
        pin_coords.append(
        (
            math.floor(center + radius * math.cos(angle)),
            math.floor(center + radius * math.sin(angle)),
        )
    )
    
# Helper function to map a line to Radon space
    def line_to_radon_weight(pin1, pin2):
    # Compute angle of the line
        x0, y0 = pin1
        x1, y1 = pin2
        angle = (np.arctan2(y1 - y0, x1 - x0) * 180 / np.pi) % 180

    # Closest angle index in the sinogram
        angle_idx = np.argmin(np.abs(theta - angle))

    # Compute distance from the center
        mid_x = (x0 + x1) / 2 - center
        mid_y = (y0 + y1) / 2 - center
        distance = np.sqrt(mid_x ** 2 + mid_y ** 2)

    # Scale the distance
        distance_scaled = int(distance * distance_scale_factor)
        if not (0 <= distance_scaled < sinogram.shape[0]):
            return 0  # Default weight for out-of-bound lines

    # Return the corresponding Radon value
        return sinogram[distance_scaled, angle_idx]

# Precompute lines between pins
    line_cache_y = [None] * N_PINS * N_PINS
    line_cache_x = [None] * N_PINS * N_PINS
    line_cache_weight = [1] * N_PINS * N_PINS
    line_cache_length = [0] * N_PINS * N_PINS

    radon_weights = {}

    print("Precalculating all lines... ", end="", flush=True)
    for a in range(N_PINS):
        for b in range(a + MIN_DISTANCE, N_PINS):
            x0, y0 = pin_coords[a]
            x1, y1 = pin_coords[b]
        
            d = int(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
        
            xs = np.linspace(x0, x1, d, dtype=int)
            ys = np.linspace(y0, y1, d, dtype=int)
        
            line_cache_y[b * N_PINS + a] = ys
            line_cache_y[a * N_PINS + b] = ys
            line_cache_x[b * N_PINS + a] = xs
            line_cache_x[a * N_PINS + b] = xs
            line_cache_length[b * N_PINS + a] = d
            line_cache_length[a * N_PINS + b] = d
        
            radon_weights[(a, b)] = line_to_radon_weight(pin_coords[a], pin_coords[b])
        
    print("done")

    error = np.ones(img.shape) * 0xFF - img.copy()

    img_result = np.ones(img.shape) * 0xFF

    result = Image.new('L', (img.shape[0] * SCALE, img.shape[1] * SCALE), 0xFF)
    draw = ImageDraw.Draw(result)
    line_mask = np.zeros(img.shape, np.float64)

    line_sequence = []
    pin = 0
    line_sequence.append(pin)

    last_pins = collections.deque(maxlen=MIN_LOOP)

# Initialize SVG drawing
# svg_filename = os.path.splitext(FILENAME)[0] + "-out.svg"
# dwg = svgwrite.Drawing(svg_filename, size=(length, length))
# dwg.add(dwg.rect(insert=(0, 0), size=(length, length), fill="white"))

# path = dwg.path(d="M {} {}".format(*pin_coords[0]), stroke="black", fill="none", stroke_width="0.15px")

# Initialize previous_absdiff
    previous_absdiff = float('inf')
    increase_count = 0   
    line_number = 0     

# Main thread path calculation loop
    for l in range(MAX_LINES):
        line_number += 1 
        if l % 100 == 0:
            print("%d " % l, end='', flush=True)

            img_result = result.resize(img.shape, Image.Resampling.LANCZOS)
            img_result = np.array(img_result)

            diff = img_result - img
            mul = np.uint8(img_result < img) * 254 + 1
            absdiff = diff * mul
            current_absdiff = absdiff.sum() / (length * length)
        
            max_possible_absdiff = 255
            percentage_diff = (current_absdiff / max_possible_absdiff) * 100
            print(f"{percentage_diff:.2f}%")

            if l > 2000:
                improvement = previous_absdiff - current_absdiff
                if improvement < 1e-3:  # Define a small threshold for improvement
                    increase_count += 1
                else:
                    increase_count = 0

                if increase_count >= 3:
                    print("Breaking early due to stagnation.")
                    break

            previous_absdiff = current_absdiff

        max_score = -math.inf
        best_pin = -1

        for offset in range(MIN_DISTANCE, N_PINS - MIN_DISTANCE):
            test_pin = (pin + offset) % N_PINS
            if test_pin in last_pins:
                continue

            xs = line_cache_x[test_pin * N_PINS + pin]
            ys = line_cache_y[test_pin * N_PINS + pin]
            line_len = line_cache_length[test_pin * N_PINS + pin]
            line_err = np.sum(error[ys, xs]) * line_cache_weight[test_pin * N_PINS + pin]

            radon_weight = radon_weights.get((pin, test_pin), 1e-6)
            length_factor = 1.0 / line_len if line_len > 0 else 1.0

            total_score = line_err * radon_weight * length_factor

            if total_score > max_score:
                max_score = total_score
                best_pin = test_pin

        line_sequence.append(best_pin)

        xs = line_cache_x[best_pin * N_PINS + pin]
        ys = line_cache_y[best_pin * N_PINS + pin]
        weight = LINE_WEIGHT * line_cache_weight[best_pin * N_PINS + pin]

    # path.push("L {} {}".format(*pin_coords[best_pin]))
        line_mask.fill(0)
        line_mask[ys, xs] = weight
        error -= line_mask
        error.clip(0, 255)

        draw.line(
        [(pin_coords[pin][0] * SCALE, pin_coords[pin][1] * SCALE),
         (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE)],
        fill=f"{LINE_COLOR}", width=1)

        last_pins.append(best_pin)
        pin = best_pin
    return length,result,line_number,current_absdiff

def main():
    # Create a Tkinter root window (it will not be shown)
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog to select the file
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )

    SET_LINES = 0

    N_PINS = 36 * 8  # Number of pins
    MIN_LOOP = 1  # Minimum loop before it returns to the same pin
    MIN_DISTANCE = 3 # Minimum distance between pins 
    LINE_WEIGHT = 18  # Line weight (thickness) more = darker
    FILENAME = file_path  # File path of the image
    SCALE = 6  # Scale factor it wll revert back to 1024 x 1024 once it is done
    SHARP = .8  # Sharpness enhancement factor
    BRIGHT = 0.8  # Brightness enhancement factor
    CONTRAST = 1  # Contrast enhancement factor
    LINE_COLOR = "black"  # Line color  
    
    if SET_LINES != 0:
        MAX_LINES = SET_LINES
    else:
        MAX_LINES = int(((N_PINS * (N_PINS - 1)) // 2))

    tic = time.perf_counter()

    # Load and preprocess the image
    img = Image.open(FILENAME).convert("L")

    L_test = np.array(img)

    # Auto image adjustments
    average_luminance = L_test.mean() / 255.0 * 2.0
    print(f"Average luminance: {average_luminance}")
    #BRIGHT = float(f"{2.0 - average_luminance:.2f}")
    if average_luminance > 0.4: 
        LINE_WEIGHT = LINE_WEIGHT - 2 
        
    if average_luminance < 0.2:
        CONTRAST = 2.0
        LINE_WEIGHT = LINE_WEIGHT - 2
        BRIGHT = BRIGHT + 0.2
        
    if average_luminance > 1.5:
        LINE_WEIGHT = LINE_WEIGHT + 5
        CONTRAST = 2.0

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

    img = ImageOps.grayscale(new_image)

    # Increase sharpness
    sharpness_enhancer = ImageEnhance.Sharpness(img)
    img = sharpness_enhancer.enhance(SHARP)
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(CONTRAST)  
    # Increase brightness
    brightness_enhancer = ImageEnhance.Brightness(img)
    img = brightness_enhancer.enhance(BRIGHT)

    img = np.array(img)
    average_luminance = img.mean() / 255.0 * 2.0
    print(f"Average luminance: {average_luminance}")

    length, result, line_number, current_absdiff = new_func(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img, LINE_COLOR) 

    img_result = result.resize(img.shape, Image.Resampling.LANCZOS)
    img_result = np.array(img_result)

    diff = img_result - img
    mul = np.uint8(img_result < img) * 254 + 1
    absdiff = diff * mul
    absdiff = absdiff.sum() / (length * length)

    max_possible_absdiff = 255  # Maximum possible per-pixel difference
    percentage_diff = (current_absdiff / max_possible_absdiff) * 100

    # Print the percentage difference
    print(f"{percentage_diff:.2f}%")

    print('\x07')
    toc = time.perf_counter()
    print("%.1f seconds" % (toc - tic))

    result_1024 = result.resize((1024, 1024), Image.Resampling.LANCZOS)

    result.shape = (length, length)

    result_1024.save(os.path.splitext(FILENAME)[0] + f"_Diff{percentage_diff:.2f}_LW_{LINE_WEIGHT}_LT_{line_number}".replace('.', '_') + ".png")
    print(f"\nThread art completed in {toc - tic:.1f} seconds")

if __name__ == "__main__":
    main()
