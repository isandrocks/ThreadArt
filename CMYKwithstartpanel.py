from PIL import Image, ImageOps
import numpy as np
from SAapp import new_func
import os
import tkinter as tk
from tkinter import filedialog

SET_LINES = 0
N_PINS = 36 * 8
MIN_LOOP = 5
MIN_DISTANCE = 35
LINE_WEIGHT = 15
SCALE = 7
LINE_COLOR = 'black'
INVERT = True
FILE_PATH = ""

# Create the Tkinter root window
root = tk.Tk()
root.title("Edit Settings")

invert_var = tk.BooleanVar(value=INVERT)\
    
output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

def update_settings():
    global SET_LINES, N_PINS, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, LINE_COLOR, INVERT, FILE_PATH
    SET_LINES = int(set_lines_entry.get())
    N_PINS = int(n_pins_entry.get())
    MIN_LOOP = int(min_loop_entry.get())
    MIN_DISTANCE = int(min_distance_entry.get())
    LINE_WEIGHT = int(line_weight_entry.get())
    SCALE = int(scale_entry.get())
    LINE_COLOR = line_color_entry.get()
    INVERT = invert_var.get()
    FILE_PATH = file_path_entry.get()

def run_code():
    update_settings()
    run_string_art()

def display_image(image_path):
    img = Image.open(image_path)
    img.show()

def select_file():
    global FILE_PATH
    FILE_PATH = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, FILE_PATH)

def run_string_art():
    FILENAME = FILE_PATH
    if SET_LINES != 0:
        MAX_LINES = SET_LINES
    else:
        MAX_LINES = int(((N_PINS * (N_PINS - 1)) // 2))
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

    img = img.convert("RGB")
    if INVERT:
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
                N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, channel_img, LINE_COLOR
            )
            results.append(np.array(result))  # Ensure result is a numpy array
            lengths.append(line_number)
            diffs.append(current_absdiff)

        # Ensure we have exactly four channels
        if len(results) != 4:
            raise ValueError("Expected 4 channels for CMYK image, got {}".format(len(results)))

        # Merge the results into a single CMYK image
        combined_result = Image.merge("CMYK", [Image.fromarray(channel) for channel in results])

        return combined_result, lengths, diffs

    result, length, current_absdiff = string_art_cmyk(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img)
    print(f"Total lines: {sum(length)}")
    print('\x07')

    result_1024 = result.resize((1024, 1024), Image.Resampling.LANCZOS)
    result_1024 = result_1024.convert("RGB")
    if INVERT:
        result_1024 = ImageOps.invert(result_1024)

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(FILENAME))[0] + f"_LW_{LINE_WEIGHT}_{length}".replace('.', '_') + ".png")
    result_1024.save(output_path)
    display_image(output_path)

tk.Label(root, text="SET_LINES").grid(row=0, column=0)
set_lines_entry = tk.Entry(root)
set_lines_entry.insert(0, SET_LINES)
set_lines_entry.grid(row=0, column=1)

tk.Label(root, text="N_PINS").grid(row=1, column=0)
n_pins_entry = tk.Entry(root)
n_pins_entry.insert(0, N_PINS)
n_pins_entry.grid(row=1, column=1)

tk.Label(root, text="MIN_LOOP").grid(row=2, column=0)
min_loop_entry = tk.Entry(root)
min_loop_entry.insert(0, MIN_LOOP)
min_loop_entry.grid(row=2, column=1)

tk.Label(root, text="MIN_DISTANCE").grid(row=3, column=0)
min_distance_entry = tk.Entry(root)
min_distance_entry.insert(0, MIN_DISTANCE)
min_distance_entry.grid(row=3, column=1)

tk.Label(root, text="LINE_WEIGHT").grid(row=4, column=0)
line_weight_entry = tk.Entry(root)
line_weight_entry.insert(0, LINE_WEIGHT)
line_weight_entry.grid(row=4, column=1)

tk.Label(root, text="SCALE").grid(row=5, column=0)
scale_entry = tk.Entry(root)
scale_entry.insert(0, SCALE)
scale_entry.grid(row=5, column=1)

tk.Label(root, text="LINE_COLOR").grid(row=6, column=0)
line_color_entry = tk.Entry(root)
line_color_entry.insert(0, LINE_COLOR)
line_color_entry.grid(row=6, column=1)

tk.Label(root, text="INVERT").grid(row=7, column=0)
invert_check = tk.Checkbutton(root, variable=invert_var)
invert_check.grid(row=7, column=1)

tk.Label(root, text="FILE_PATH").grid(row=8, column=0)
file_path_entry = tk.Entry(root)
file_path_entry.insert(0, FILE_PATH)
file_path_entry.grid(row=8, column=1)
tk.Button(root, text="Browse", command=select_file).grid(row=8, column=2)

tk.Button(root, text="Apply", command=update_settings).grid(row=9, columnspan=3)
tk.Button(root, text="Run Code", command=run_code).grid(row=10, columnspan=3)

root.mainloop()