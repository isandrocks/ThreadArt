from PIL import Image, ImageOps, ImageTk
import numpy as np
from SAapp import string_art
import os
import tkinter as tk
from tkinter import filedialog
import threading
import queue
import sys

SET_LINES = 0
N_PINS = 36 * 8
MIN_LOOP = 2
MIN_DISTANCE = 3
LINE_WEIGHT = 15
SCALE = 5
LINE_COLOR = 'black'
INVERT = True
FILE_PATH = ""
GRAYSCALE = False

# Create the Tkinter root window
root = tk.Tk()
root.title("Edit Settings")
root.attributes("-topmost", True)

# Tkinter window colors and theme
TK_BG = '#272727'
TK_FG = '#d1d1d1'
TK_SEL_BG = '#464646'
padding_options = {'padx': 10, 'pady': 5}

root.configure(bg=TK_BG)

# Create a Label widget to display the image
image_label = tk.Label(root)
image_label.grid(row=11, columnspan=3, **padding_options)

invert_var = tk.BooleanVar(value=INVERT)
grayscale_var = tk.BooleanVar(value=GRAYSCALE)

output_dir = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(output_dir, exist_ok=True)

# Create a queue to communicate between the main thread and the worker thread
result_queue = queue.Queue()

def update_settings():
    global SET_LINES, N_PINS, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, LINE_COLOR, INVERT, FILE_PATH, GRAYSCALE
    SET_LINES = int(set_lines_entry.get())
    N_PINS = int(n_pins_entry.get())
    MIN_LOOP = int(min_loop_entry.get())
    MIN_DISTANCE = int(min_distance_entry.get())
    LINE_WEIGHT = int(line_weight_entry.get())
    SCALE = int(scale_entry.get())
    INVERT = invert_var.get()
    FILE_PATH = file_path_entry.get()
    GRAYSCALE = grayscale_var.get()

def run_code():
    update_settings()
    threading.Thread(target=run_string_art).start()
    root.after(100, check_queue)

def check_queue():
    try:
        result = result_queue.get_nowait()
        display_image(result)
    except queue.Empty:
        root.after(100, check_queue)

def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((512, 512), Image.Resampling.LANCZOS)  # Resize the image to fit the window
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk  

def select_file():
    global FILE_PATH
    FILE_PATH = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, FILE_PATH)

def percent_diff(absdiff):
    max_possible_absdiff = 255  # Maximum possible per-pixel difference
    percentage_diff = (absdiff / max_possible_absdiff) * 100
    return percentage_diff    

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

    if GRAYSCALE:
        result, length, current_absdiff = string_art_grayscale(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img)
        length = [length]  # Ensure length is a list
    else:
        result, length, current_absdiff = string_art_cmyk(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img)

    print(f"Total lines: {sum(length)}")
    
    if GRAYSCALE:
        avg_absdiff = percent_diff(current_absdiff)
        print(f"Average error: {avg_absdiff:.2f}%")
    else:
        C_absdiff = percent_diff(current_absdiff[0])
        M_absdiff = percent_diff(current_absdiff[1])
        Y_absdiff = percent_diff(current_absdiff[2])
        K_absdiff = percent_diff(current_absdiff[3])
        avg_absdiff = (C_absdiff + M_absdiff + Y_absdiff + K_absdiff) / 4
        print(f"Average error: {avg_absdiff:.2f}%")
    
    print('\x07')

    result_1024 = result.resize((1024, 1024), Image.Resampling.LANCZOS)
    result_1024 = result_1024.convert("RGB")
    if INVERT:
        result_1024 = ImageOps.invert(result_1024)

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(FILENAME))[0] + f"_S_{SCALE}_LW_{LINE_WEIGHT}_{sum(length)}".replace('.', '_') + ".png")
    result_1024.save(output_path)
    result_queue.put(output_path)
    print(f"Saved result to {output_path}")

def string_art_grayscale(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img):
    # Convert image to grayscale
    img_gray = img.convert("L")
    gray_channel = np.array(img_gray)
    
    print("Processing grayscale channel...")
    length, result, line_number, current_absdiff = string_art(
        N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, gray_channel, LINE_COLOR
    )
    
    result_img = Image.fromarray(np.array(result))
    return result_img, line_number, current_absdiff

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
        length, result, line_number, current_absdiff = string_art(
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

tk.Label(root, text="SET_LINES", bg=TK_BG, fg=TK_FG).grid(row=0, column=0)
set_lines_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG)
set_lines_entry.insert(0, SET_LINES)
set_lines_entry.grid(row=0, column=1)

tk.Label(root, text="N_PINS", bg=TK_BG, fg=TK_FG).grid(row=1, column=0)
n_pins_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG)
n_pins_entry.insert(0, N_PINS)
n_pins_entry.grid(row=1, column=1)

tk.Label(root, text="MIN_LOOP", bg=TK_BG, fg=TK_FG).grid(row=2, column=0)
min_loop_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG)
min_loop_entry.insert(0, MIN_LOOP)
min_loop_entry.grid(row=2, column=1)

tk.Label(root, text="MIN_DISTANCE", bg=TK_BG, fg=TK_FG).grid(row=3, column=0)
min_distance_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG)
min_distance_entry.insert(0, MIN_DISTANCE)
min_distance_entry.grid(row=3, column=1)

tk.Label(root, text="LINE_WEIGHT", bg=TK_BG, fg=TK_FG).grid(row=4, column=0)
line_weight_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG)
line_weight_entry.insert(0, LINE_WEIGHT)
line_weight_entry.grid(row=4, column=1)

tk.Label(root, text="SCALE", bg=TK_BG, fg=TK_FG).grid(row=5, column=0)
scale_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG)
scale_entry.insert(0, SCALE)
scale_entry.grid(row=5, column=1)

tk.Label(root, text="GRAYSCALE", bg=TK_BG, fg=TK_FG).grid(row=6, column=0)
grayscale_check = tk.Checkbutton(root, variable=grayscale_var, bg=TK_BG, fg=TK_FG, selectcolor=TK_SEL_BG)
grayscale_check.grid(row=6, column=1)

tk.Label(root, text="INVERT", bg=TK_BG, fg=TK_FG).grid(row=7, column=0)
invert_check = tk.Checkbutton(root, variable=invert_var, bg=TK_BG, fg=TK_FG, selectcolor=TK_SEL_BG)
invert_check.grid(row=7, column=1)

tk.Label(root, text="FILE_PATH", bg=TK_BG, fg=TK_FG).grid(row=8, column=0, **padding_options)
file_path_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG)
file_path_entry.insert(0, FILE_PATH)
file_path_entry.grid(row=8, column=1)
tk.Button(root, text="Browse", command=select_file, bg=TK_SEL_BG, fg=TK_FG).grid(row=8, column=2, **padding_options)

tk.Button(root, text="Apply", command=update_settings, bg=TK_SEL_BG, fg=TK_FG).grid(row=9, columnspan=3)
tk.Button(root, text="Run Code", command=run_code, bg=TK_SEL_BG, fg=TK_FG).grid(row=10, columnspan=3)

# Create a Text widget to display terminal prints
output_text = tk.Text(root, bg=TK_SEL_BG, fg=TK_FG, wrap='word', height=5)
output_text.grid(row=12, columnspan=3, **padding_options)

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

sys.stdout = StdoutRedirector(output_text)

root.mainloop()