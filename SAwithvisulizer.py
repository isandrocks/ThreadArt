import collections
import math
import os
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import time
import svgwrite
import tkinter as tk
from tkinter import filedialog
import gc 
from moviepy import ImageSequenceClip
from tqdm import tqdm

def string_art(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, FILENAME, SCALE, SAVE_SVG, SAVE_JSON, img, output_dir):
    
    assert img.shape[0] == img.shape[1]
    length = img.shape[0]
    
    # Cut away everything around a central circle
    X, Y = np.ogrid[0:length, 0:length]
    circlemask = (X - length/2) ** 2 + (Y - length/2) ** 2 > length/2 * length/2
    img[circlemask] = 0xFF

    # Precalculate the coordinates of every pin    
    pin_coords = []
    center = length / 2
    radius = length / 2 - 1/2    
    
    for i in range(N_PINS):
        angle = 2 * math.pi * i / N_PINS
        pin_coords.append((math.floor(center + radius * math.cos(angle)),
                       math.floor(center + radius * math.sin(angle))))

    line_cache_y = [None] * N_PINS * N_PINS
    line_cache_x = [None] * N_PINS * N_PINS
    line_cache_weight = [1] * N_PINS * N_PINS # Turned out to be unnecessary, unused
    line_cache_length = [0] * N_PINS * N_PINS

    print("Precalculating all lines... ", end='', flush=True)

    for a in range(N_PINS):
        for b in range(a + MIN_DISTANCE, N_PINS):
            x0 = pin_coords[a][0]
            y0 = pin_coords[a][1]

            x1 = pin_coords[b][0]
            y1 = pin_coords[b][1]

            d = int(math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0)*(y1 - y0)))

            xs = np.linspace(x0, x1, d, dtype=int)
            ys = np.linspace(y0, y1, d, dtype=int)

            line_cache_y[b*N_PINS + a] = ys
            line_cache_y[a*N_PINS + b] = ys
            line_cache_x[b*N_PINS + a] = xs
            line_cache_x[a*N_PINS + b] = xs
            line_cache_length[b*N_PINS + a] = d
            line_cache_length[a*N_PINS + b] = d

    print("done")

    # Initialize SVG drawing
    if SAVE_SVG:
        svg_filename = os.path.join(output_dir, os.path.splitext(os.path.basename(FILENAME))[0] + "-out.svg")
        dwg = svgwrite.Drawing(svg_filename, size=(img.shape[0] * SCALE, img.shape[1] * SCALE))
        dwg.add(dwg.rect(insert=(0, 0), size=(img.shape[0] * SCALE, img.shape[1] * SCALE), fill="white"))
        path = dwg.path(d="M {} {}".format(*pin_coords[0]), stroke="black", fill="none", stroke_width="0.15px")

    # Initialize variables for the calculation loop
    error = np.ones(img.shape) * 0xFF - img.copy()
    img_result = np.ones(img.shape) * 0xFF
    result = Image.new('L', (img.shape[0] * SCALE, img.shape[1] * SCALE), 0xFF)
    draw = ImageDraw.Draw(result)
    line_mask = np.zeros(img.shape, np.float64)
    line_sequence = []
    pin = 0
    line_sequence.append(pin)
    last_pins = collections.deque(maxlen = MIN_LOOP)    
    frames = []
    increase_count = 0
    previous_absdiff = float('inf')
    line_number = 0     

    # Main thread path calculation loop
    for l in range(MAX_LINES):
        line_number += 1 
        
        #check for differance between the original image and the current image
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

            #break out of the loop if the difference is less than 1e-3
            if l > 2000:
                improvement = previous_absdiff - current_absdiff
                if improvement < 1e-3:
                    increase_count += 1
                else:
                    increase_count = 0

                if increase_count >= 2:
                    print("Breaking early due to stagnation.")
                    break

                previous_absdiff = current_absdiff
                gc.collect()
       
        max_err = -math.inf
        best_pin = -1

        for offset in range(MIN_DISTANCE, N_PINS - MIN_DISTANCE):
            test_pin = (pin + offset) % N_PINS
            if test_pin in last_pins:
                continue

            xs = line_cache_x[test_pin * N_PINS + pin]
            ys = line_cache_y[test_pin * N_PINS + pin]
            line_err = np.sum(error[ys,xs]) * line_cache_weight[test_pin*N_PINS + pin]

            if line_err > max_err:
                max_err = line_err
                best_pin = test_pin

        xs = line_cache_x[best_pin * N_PINS + pin]
        ys = line_cache_y[best_pin * N_PINS + pin]
        weight = LINE_WEIGHT * line_cache_weight[best_pin*N_PINS + pin]

        line_mask.fill(0)
        line_mask[ys, xs] = weight
        error[ys, xs] -= line_mask[ys, xs]
        error.clip(0, 255, out=error)
        
        # Collect the various image and line data
        #SVG data
        if SAVE_SVG:
            path.push("L {} {}".format(*pin_coords[best_pin]))
        
        #Json data 
        if SAVE_JSON:  
            line_sequence.append(best_pin)            
        
        #Png data    
        draw.line(
        [(pin_coords[pin][0] * SCALE, pin_coords[pin][1] * SCALE),
        (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE)],
        fill=0, width=1)

        #video data
        line_segment = [
            (pin_coords[pin][0] * SCALE, pin_coords[pin][1] * SCALE),
            (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE)
            ]           
        frames.append(line_segment)
        
        #clean up and set variables before the next iteration
        x0 = pin_coords[pin][0]
        y0 = pin_coords[pin][1]

        x1 = pin_coords[best_pin][0]
        y1 = pin_coords[best_pin][1]

        last_pins.append(best_pin)
        pin = best_pin
        
    if SAVE_SVG:
        dwg.add(path)
        dwg.save()
        
    return length,result,line_sequence,frames,line_number,current_absdiff

def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog to select the file
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )

    SET_LINES = 0 # Set to 0 to use default value
    N_PINS = 36 * 8
    MIN_LOOP = 1
    MIN_DISTANCE = 2
    LINE_WEIGHT = 17
    SCALE = 7
    SAVE_SVG = False
    SAVE_JSON = False
    
    FILENAME = file_path    
    if SET_LINES != 0:
        MAX_LINES = SET_LINES
    else:
        MAX_LINES = int(((N_PINS * (N_PINS - 1)) // 2))

    print("max lines: %d" % MAX_LINES)

    tic = time.perf_counter()
    image = Image.open(FILENAME)
    width, height = image.size

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

    resized_image = image.resize((new_width, new_height))

    if resized_image.size[0] != resized_image.size[1]:
        new_image = resized_image.crop((new_width // 2 - 256, new_height // 2 - 256, new_width // 2 + 256, new_height // 2 + 256))
    else:
        new_image = resized_image

    img = new_image.convert('L')
    img = ImageOps.grayscale(img)
    img = np.array(img)
    
    #start the string art function
    length, result, line_sequence, frames, line_number, current_absdiff = string_art(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, FILENAME, SCALE, SAVE_SVG, SAVE_JSON, img, output_dir)
    
    #start of the processing for saving
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

    # Save the final image
    result_1024 = result.resize((1024, 1024), Image.Resampling.LANCZOS)
    result_1024.save(os.path.join(output_dir, os.path.splitext(os.path.basename(FILENAME))[0] + f"_LW_{LINE_WEIGHT}".replace('.', '_') + ".png"))

    # Save the line sequence as a JSON file
    if SAVE_JSON:
        with open(os.path.join(output_dir, os.path.splitext(os.path.basename(FILENAME))[0] + ".json"), "w") as f:
            f.write(str(line_sequence))

    print("creating video frames...")
    
    frame = Image.new('L', (img.shape[0] * SCALE, img.shape[1] * SCALE), 0xFF)
    
    def reconstruct_frame(lines):
        draw = ImageDraw.Draw(frame)
        draw.line(lines, fill='black', width=1)
        return frame

    video_frames = []
    
    for frame_data in tqdm(frames, desc="Processing frames"):
        video_frame = reconstruct_frame(frame_data)
        resized_frame = video_frame.resize((512, 512), Image.Resampling.LANCZOS).convert('RGB')        
        video_frames.append(resized_frame)

    # Save the frames as an MP4 video
    clip = ImageSequenceClip([np.array(frame) for frame in video_frames], fps=(line_number / 17))  # Adjust fps as needed
    clip.write_videofile(os.path.join(output_dir, os.path.splitext(os.path.basename(FILENAME))[0] + "-out.mp4"), codec='libx264')

if __name__ == "__main__":
    main()