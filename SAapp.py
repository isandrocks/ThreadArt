from skimage.transform import radon
import collections
import math
import os
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
import random
from scipy.ndimage import gaussian_filter


def string_art(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img):
    assert img.shape[0] == img.shape[1]
    length = img.shape[0]

    # Apply circular mask
    X, Y = np.ogrid[0:length, 0:length]
    circlemask = (X - length / 2) ** 2 + (Y - length / 2) ** 2 > (length / 2) ** 2
    img[circlemask] = 0xFF

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

    error = np.ones(img.shape) * 0xFF - img.copy()

    # Compute the Radon Transform of the image
    theta = np.linspace(0.0, 180.0, max(img.shape), endpoint=False)
    sinogram = radon(error, theta=theta, preserve_range=True)

    # Normalize the sinogram for line weighting
    sinogram /= sinogram.max()

    # Precompute the scaling factor for distance
    distance_scale_factor = sinogram.shape[0] / (length / 2)

    # Helper function to map a line to Radon space
    def line_to_radon_weight(pin1, pin2):
        x0, y0 = pin1
        x1, y1 = pin2
        angle = (np.arctan2(y1 - y0, x1 - x0) * 180 / np.pi) % 180

        angle_idx = np.argmin(np.abs(theta - angle))

        mid_x = (x0 + x1) / 2 - center
        mid_y = (y0 + y1) / 2 - center
        distance = np.sqrt(mid_x**2 + mid_y**2)

        distance_scaled = int((distance * distance_scale_factor) - 1)

        return sinogram[distance_scaled, angle_idx]

    print("Precalculating all lines... ", end="", flush=True)

    # Precompute lines between pins
    line_cache_y = [None] * N_PINS * N_PINS
    line_cache_x = [None] * N_PINS * N_PINS
    line_cache_weight = [1] * N_PINS * N_PINS
    line_cache_length = [0] * N_PINS * N_PINS
    radon_weights = {}

    for a in range(N_PINS):
        for b in range(a + MIN_DISTANCE, N_PINS):
            x0, y0 = pin_coords[a]
            x1, y1 = pin_coords[b]

            d = int(math.sqrt((x1 - x0) ** 2 + (y0 - y1) ** 2))

            xs = np.linspace(x0, x1, d, dtype=int)
            ys = np.linspace(y0, y1, d, dtype=int)

            # Store the calculated values in the cache
            line_cache_y[b * N_PINS + a] = ys
            line_cache_y[a * N_PINS + b] = ys
            line_cache_x[b * N_PINS + a] = xs
            line_cache_x[a * N_PINS + b] = xs
            line_cache_length[b * N_PINS + a] = d
            line_cache_length[a * N_PINS + b] = d

            radon_weights[(a, b)] = line_to_radon_weight(pin_coords[a], pin_coords[b])

    print("done")

    def find_opposite_pin(pin, N_PINS):
        return (pin + N_PINS // 2) % N_PINS

    # Initialize variables for the calculation loop
    img_result = np.ones(img.shape) * 0xFF
    result = Image.new("L", (img.shape[0] * SCALE, img.shape[1] * SCALE), 0xFF)
    draw = ImageDraw.Draw(result)
    line_mask = np.zeros(img.shape, np.float64)
    last_pins = collections.deque(maxlen=MIN_LOOP)
    last_pincords = collections.deque(maxlen=(MIN_LOOP + 20))
    previous_absdiff = float("inf")
    increase_count = 0
    line_number = 0
    frames = []
    pin_sequence = []
    pin = 0
    op_pin_count = 0
    opc_error = []
    last_p_count = 0
    break_outer_loop = False  # Flag to break out of the outer loop

    # Main calculation loop
    for l in range(MAX_LINES):
        if break_outer_loop:
            break  # Break out of the outer loop if the flag is set
        line_number += 1

        # check for differance between the original image and the current image
        if l % 100 == 0:
            opc_error.append(op_pin_count)
            if sum(opc_error) >= (N_PINS / 2):
                print("Breaking early due to cross center stagnation.")
                break
            op_pin_count = 0

            img_result = result.resize(img.shape, Image.Resampling.LANCZOS)
            img_result = np.array(img_result)

            diff = img_result - img
            mul = np.uint8(img_result < img) * 254 + 1
            absdiff = diff * mul
            current_absdiff = absdiff.sum() / (length * length)

            max_possible_absdiff = 255
            percentage_diff = (current_absdiff / max_possible_absdiff) * 100
            print(f"{l} {percentage_diff:.2f}%")

            # break out of the loop if the difference is less than 1e-3
            if l > 1000:
                improvement = previous_absdiff - current_absdiff
                if improvement < 1e-3:
                    increase_count += 1
                else:
                    increase_count = 0

                if increase_count >= 3:
                    print("Breaking early due to stagnation.")
                    break

            previous_absdiff = current_absdiff

        max_score = -math.inf
        best_pin = -1

        offsets = list(range(MIN_DISTANCE, N_PINS - MIN_DISTANCE))
        random.shuffle(offsets)

        for offset in offsets:

            test_pin = (pin + offset) % N_PINS

            if test_pin in last_pins:
                continue

            xs = line_cache_x[test_pin * N_PINS + pin]
            ys = line_cache_y[test_pin * N_PINS + pin]

            line_err = np.sum(error[ys, xs]) * line_cache_weight[test_pin * N_PINS + pin]

            radon_weight = radon_weights.get((pin, test_pin))
            if radon_weight is None:
                radon_weight = line_to_radon_weight(pin_coords[pin], pin_coords[test_pin])
                radon_weights[(pin, test_pin)] = radon_weight

            total_score = line_err * radon_weight
            op_pin = find_opposite_pin(pin, N_PINS)

            if total_score > max_score:
                last_pincords.append(
                    [
                        (pin_coords[test_pin][0] * SCALE, pin_coords[test_pin][1] * SCALE),
                        (pin_coords[pin][0] * SCALE, pin_coords[pin][1] * SCALE),
                    ]
                )
                current_pincords = [
                    (pin_coords[pin][0] * SCALE, pin_coords[pin][1] * SCALE),
                    (pin_coords[test_pin][0] * SCALE, pin_coords[test_pin][1] * SCALE),
                ]
                if current_pincords in last_pincords or test_pin == op_pin:
                    if l > 2000:
                        op_pin_count += 1
                        last_p_count += 1
                        if op_pin_count > (N_PINS / 8) or last_p_count > 4:
                            print("Breaking early due to stagnation. Repeating pin cords")
                            break_outer_loop = True  # Set the flag to break out of the outer loop
                            break
                else:
                    last_p_count = 0

                last_pincords.append(current_pincords)
                max_score = total_score
                best_pin = test_pin

        xs = line_cache_x[best_pin * N_PINS + pin]
        ys = line_cache_y[best_pin * N_PINS + pin]
        weight = LINE_WEIGHT * line_cache_weight[best_pin * N_PINS + pin]

        line_mask.fill(0)
        line_mask[ys, xs] = weight

        error -= line_mask
        error.clip(0, 255)

        # image data
        draw.line(
            [
                (pin_coords[pin][0] * SCALE, pin_coords[pin][1] * SCALE),
                (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE),
            ],
            fill=0,
            width=1,
        )

        # frame data
        line_segment = [
            (pin_coords[pin][0] * SCALE, pin_coords[pin][1] * SCALE),
            (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE),
        ]

        frames.append(line_segment)

        last_pins.append(best_pin)
        pin_sequence.append(best_pin)
        pin = best_pin

    return pin_sequence, result, line_number, current_absdiff, frames


def main():

    # Create a Tkinter root window (it will not be shown)
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog to select the file
    file_path = filedialog.askopenfilename(
        title="Select an image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )

    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    SET_LINES = 0

    N_PINS = 36 * 8  # Number of pins
    MIN_LOOP = 1  # Minimum loop before it returns to the same pin
    MIN_DISTANCE = 3  # Minimum distance between pins
    LINE_WEIGHT = 40  # Line weight (thickness) more = darker
    FILENAME = file_path  # File path of the image
    SCALE = 4  # Scale factor it wll revert back to 1024 x 1024 once it is done

    if SET_LINES != 0:
        MAX_LINES = SET_LINES
    else:
        MAX_LINES = int(((N_PINS * (N_PINS - 1)) // 2))

    tic = time.perf_counter()

    # Load and preprocess the image
    img = Image.open(FILENAME).convert("L")

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
        new_image = resized_image.crop(
            (new_width // 2 - 256, new_height // 2 - 256, new_width // 2 + 256, new_height // 2 + 256)
        )
    else:
        new_image = resized_image

    img = ImageOps.grayscale(new_image)

    img = np.array(img)

    pin_sequence, result, line_number, current_absdiff, frames = string_art(
        N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img
    )

    img_result = result.resize(img.shape, Image.Resampling.LANCZOS)
    img_result = np.array(img_result)

    max_possible_absdiff = 255  # Maximum possible per-pixel difference
    percentage_diff = (current_absdiff / max_possible_absdiff) * 100

    # Print the percentage difference
    print(f"{percentage_diff:.2f}%")

    print("\x07")
    toc = time.perf_counter()
    print("%.1f seconds" % (toc - tic))

    result_1024 = result.resize((1024, 1024), Image.Resampling.LANCZOS)
    result_1024.save(
        os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(FILENAME))[0] + f"_LW_{LINE_WEIGHT}".replace(".", "_") + ".png",
        )
    )

    with open(os.path.join(output_dir, os.path.splitext(os.path.basename(FILENAME))[0] + ".json"), "w") as f:
        f.write(str(pin_sequence))


if __name__ == "__main__":
    main()
