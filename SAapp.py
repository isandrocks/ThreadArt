import collections
import math
import os
from PIL import Image, ImageOps, ImageDraw, ImageChops
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog
import random
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim


def string_art_multiscale(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img, edge_map=None, EDGE_BOOST=2.0, SSIM_TARGET=0.65, no_stagnation=False):
    # Pass 1: Coarse
    coarse_pins = N_PINS // 3
    img_coarse = Image.fromarray(img).resize((128, 128), Image.Resampling.LANCZOS)
    img_coarse_np = np.array(img_coarse)
    
    print("--- MULTISCALE: COARSE PASS ---")
    seq_c, res_c, ln_c, diff_c, frames_c = string_art(
        coarse_pins, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img_coarse_np,
        edge_map=None, EDGE_BOOST=EDGE_BOOST, SSIM_TARGET=0.45, no_stagnation=no_stagnation
    )
    
    # Scale up coarse result to original size
    length = img.shape[0]
    res_c_scaled = res_c.resize((length, length), Image.Resampling.LANCZOS)
    res_c_np = np.array(res_c_scaled, dtype=np.float64)
    
    # Compute residual: what's left to draw
    residual_error = np.clip(res_c_np - img.astype(np.float64), 0, 255)
    target_img = np.clip(255 - residual_error, 0, 255).astype(np.uint8)
    
    print("--- MULTISCALE: FINE PASS ---")
    seq_f, res_f, ln_f, diff_f, frames_f = string_art(
        N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, target_img,
        edge_map=edge_map, EDGE_BOOST=EDGE_BOOST, SSIM_TARGET=SSIM_TARGET, no_stagnation=no_stagnation
    )
    
    # Combine results
    ratio = N_PINS / coarse_pins
    mapped_seq_c = [int(p * ratio) for p in seq_c]
    final_seq = mapped_seq_c + seq_f
    
    scale_factor = length / 128.0
    mapped_frames_c = []
    for f in frames_c:
        mapped_frames_c.append([
            (f[0][0] * scale_factor, f[0][1] * scale_factor),
            (f[1][0] * scale_factor, f[1][1] * scale_factor)
        ])
    final_frames = mapped_frames_c + frames_f
    
    res_c_full = res_c.resize((length * SCALE, length * SCALE), Image.Resampling.LANCZOS)
    final_result = ImageChops.darker(res_c_full.convert("L"), res_f.convert("L"))
    
    return final_seq, final_result, ln_c + ln_f, diff_f, final_frames

def string_art(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img, edge_map=None, EDGE_BOOST=2.0, SSIM_TARGET=0.65, no_stagnation=False):
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

    error = np.ones(img.shape, dtype=np.float64) * 0xFF - img.astype(np.float64)

    if edge_map is None:
        edge_map = sobel(img.astype(np.float64))
        if edge_map.max() > 0:
            edge_map /= edge_map.max()

    print("Precalculating all lines... ", end="", flush=True)

    # Precompute lines between pins
    line_cache_y = [None] * N_PINS * N_PINS
    line_cache_x = [None] * N_PINS * N_PINS
    line_cache_length = [0] * N_PINS * N_PINS

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

    print("done")

    # Initialize variables for the calculation loop
    result = Image.new("L", (img.shape[0] * SCALE, img.shape[1] * SCALE), 0xFF)
    draw = ImageDraw.Draw(result)
    line_mask = np.zeros(img.shape, np.float64)
    last_pins = collections.deque(maxlen=MIN_LOOP)
    line_number = 0
    frames = []
    pin_sequence = []
    pin = 0
    current_absdiff = 0.0
    error_history = []  # rolling window for stagnation detection

    # Main calculation loop
    for l in range(MAX_LINES):
        line_number += 1

        # ---- Stagnation & SSIM check every 100 lines ----
        if l % 100 == 0:
            from scipy.ndimage import gaussian_filter
            img_result = result.resize((length, length), Image.Resampling.LANCZOS)
            img_result_np = np.array(img_result, dtype=np.float64)

            # Invert and blur to match string art perceptual structure to the 0.5-0.9 target scale
            b_img = gaussian_filter(255.0 - img.astype(np.float64), sigma=2.0)
            b_res = gaussian_filter(255.0 - img_result_np, sigma=2.0)

            # SSIM target check
            current_ssim = ssim(b_img, b_res, data_range=255.0)
            print(f"{l} SSIM: {current_ssim:.4f}")

            if current_ssim > SSIM_TARGET:
                print("Breaking early: SSIM target reached.")
                break

            error_history.append(current_ssim)

            if not no_stagnation and len(error_history) > 5:
                # Compare current vs 500 lines ago (index -6)
                improvement = current_ssim - error_history[-6]
                if improvement < 0.005 and l > 500:
                    print("Breaking early due to SSIM stagnation.")
                    break
        # ---- greedy pin selection: pick pin with highest error reduction ----
        max_score = -math.inf
        best_pin = -1

        for offset in range(MIN_DISTANCE, N_PINS - MIN_DISTANCE):
            test_pin = (pin + offset) % N_PINS

            if test_pin in last_pins:
                continue

            xs = line_cache_x[test_pin * N_PINS + pin]
            ys = line_cache_y[test_pin * N_PINS + pin]

            if xs is None or ys is None:
                continue

            line_err = float(np.sum(np.minimum(error[ys, xs], LINE_WEIGHT) * (1.0 + EDGE_BOOST * edge_map[ys, xs])))

            if line_err > max_score:
                max_score = line_err
                best_pin = test_pin

        if best_pin == -1:
            break

        xs = line_cache_x[best_pin * N_PINS + pin]
        ys = line_cache_y[best_pin * N_PINS + pin]

        line_mask.fill(0)
        line_mask[ys, xs] = LINE_WEIGHT

        error -= line_mask
        np.clip(error, 0, 255, out=error)

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

    # Final absdiff computation
    img_result = result.resize((length, length), Image.Resampling.LANCZOS)
    img_result = np.array(img_result, dtype=np.float64)
    diff = np.abs(img_result - img.astype(np.float64))
    current_absdiff = diff.sum() / (length * length)

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
