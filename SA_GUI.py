from PIL import Image, ImageOps, ImageTk, ImageDraw, ImageChops
import numpy as np
from SAapp import string_art
import os
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import queue
from moviepy import ImageSequenceClip
import contextlib
from tqdm import tqdm

# Default settings
SET_LINES = 0
N_PINS = 36 * 8
MIN_LOOP = 5
MIN_DISTANCE = 5
LINE_WEIGHT = 25
SCALE = 7
INVERT = False
FILE_PATH = ""
GRAYSCALE = True
SAVE_MP4 = False
SAVE_JSON = False

# Tkinter root window
root = tk.Tk()
root.title("Edit Settings")
root.geometry("+0+0")

invert_var = tk.BooleanVar(value=INVERT)
grayscale_var = tk.BooleanVar(value=GRAYSCALE)
mp4_var = tk.BooleanVar(value=SAVE_MP4)
json_var = tk.BooleanVar(value=SAVE_JSON)

output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)

# queue to communicate between the main thread and the worker thread
result_queue = queue.Queue()


class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None

        # Bind events to show and hide the tooltip
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        
        # Get the position of the widget
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        # Create the tooltip window
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)  # Remove window borders
        self.tip_window.wm_geometry(f"+{x}+{y}")

        # Create the label for the tooltip
        label = tk.Label(
            self.tip_window,
            text=self.text,
            fg="#d1d1d1",
            bg="#464646",
            relief="solid",
            borderwidth=1,
            font=("Arial", 10, "normal"),
        )
        label.pack()

    def hide_tip(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


def find_time(seconds):
    minutes = round(seconds // 60)
    seconds = round(seconds % 60)
    return minutes, seconds


def update_settings():
    global SET_LINES, N_PINS, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, INVERT, FILE_PATH, GRAYSCALE, SAVE_MP4, SAVE_JSON
    SET_LINES = int(set_lines_entry.get())
    N_PINS = int(n_pins_entry.get())
    MIN_LOOP = int(min_loop_slider.get())
    MIN_DISTANCE = int(min_distance_slider.get())
    LINE_WEIGHT = int(line_weight_slider.get())
    SCALE = int(scale_slider.get())
    INVERT = invert_var.get()
    FILE_PATH = file_path_entry.get()
    GRAYSCALE = grayscale_var.get()
    SAVE_MP4 = mp4_var.get()
    SAVE_JSON = json_var.get()


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
    global output_path
    FILE_PATH = filedialog.askopenfilename(
        title="Select an image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(FILE_PATH))[0])
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
        MAX_LINES = int(((N_PINS * (N_PINS - 1)) // 2) / 2)
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
        new_image = resized_image.crop(
            (new_width // 2 - 256, new_height // 2 - 256, new_width // 2 + 256, new_height // 2 + 256)
        )
    else:
        new_image = resized_image

    img = new_image

    img = img.convert("RGB")
    if INVERT:
        img = ImageOps.invert(img)

    if GRAYSCALE:
        result, length, current_absdiff = string_art_grayscale(
            N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img
        )
        length = [length]  # Ensure length is a list
    else:
        result, length, current_absdiff = string_art_cmyk(
                N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img
            )


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

    print("\x07")

    result_1024 = result.resize((1024, 1024), Image.Resampling.LANCZOS)
    result_1024 = result_1024.convert("RGB")
    if INVERT:
        result_1024 = ImageOps.invert(result_1024)

    output_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(FILENAME))[0]
        + f"_S_{SCALE}_LW_{LINE_WEIGHT}_{sum(length)}".replace(".", "_")
        + ".png",
    )
    result_1024.save(output_path)
    result_queue.put(output_path)
    print(f"Saved result to {output_path}")


def string_art_grayscale(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img):
    # Convert image to grayscale
    img_gray = img.convert("L")
    img_gray = ImageOps.grayscale(img_gray)
    gray_channel = np.array(img_gray)

    with contextlib.redirect_stdout(StdoutRedirector(output_text)):
      print("Processing grayscale channel...")
      pin_sequence, result, line_number, current_absdiff, frames = string_art(
          N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, gray_channel
      )

    result_img = Image.fromarray(np.array(result))
    
      # Create video frames
    if SAVE_MP4:		
      frame = Image.new("L", (img.size[0] * SCALE, img.size[1] * SCALE), 0xFF)		
      print("Reconstructing frame data...")
      pbar_label.config(text=f"grayscale channel reconstruction...")
      progress_bar["value"] = 0
      tk_pbar = progress_bar["value"]
      progress_bar["maximum"] = len(frames)
      root.update_idletasks()
      loop_ips = 1
      frame_idx = 0
      video_frames = []

      with tqdm(total=len(frames)) as pbar:
          for frame_data in frames:
              frame_idx += 1
              draw = ImageDraw.Draw(frame)
              draw.line(frame_data, fill=0, width=1)
              resized_frame = frame.resize((512, 512), Image.Resampling.BOX).convert("RGB")
              if INVERT:
                  resized_frame = ImageOps.invert(resized_frame)
              video_frames.append(resized_frame)

              # stats update
              tk_pbar = tk_pbar + 1
              progress_bar["value"] = tk_pbar
              pbar_dict = pbar.format_dict
              loop_time = round(pbar_dict["elapsed"])
              if frame_idx % 10 == 0:
                  loop_ips = pbar_dict["rate"]
                  if loop_ips is not None:
                      loop_ips = round(loop_ips * 100)
                      loop_ips = loop_ips / 100
                  else:
                      loop_ips = 1

              start_eta = len(frames) / loop_ips
              current_eta = start_eta - loop_time
              current_minutes, current_seconds = find_time(current_eta)

              eta_label.config(
                  text=f"ETA: {current_minutes:02}:{current_seconds:02} | FPS: {loop_ips} | Frame: {frame_idx + 1}/{len(frames)}"
              )
              pbar.update(1)
              root.update_idletasks()

      # Save the frames as an MP4 video
      clip = ImageSequenceClip(
          [np.array(frame) for frame in video_frames], fps=(line_number / 17)
      )
      with contextlib.redirect_stdout(StdoutRedirector(output_text)):  
        clip.write_videofile((output_path + "_grayscale_output.mp4"), codec="libx264")

    if SAVE_JSON:
      with open((output_path + ".json"), "w") as f:
        f.write(str(pin_sequence))

    return result_img, line_number, current_absdiff


def string_art_cmyk(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img):
    # Convert image to CMYK
    img_cmyk = img.convert("CMYK")

    cmyk_channels = [np.array(img_cmyk.getchannel(channel)) for channel in range(4)]

    results = []  # To store results for each channel
    total_lines = []  # To store line counts for each channel
    diffs = []  # To store errors for each channel
    frame_data = []  # To store frame data for each channel
    frame = Image.new("CMYK", (img.size[0] * SCALE, img.size[1] * SCALE), (0, 0, 0, 0))
    trasparent_frame = Image.new("CMYK", (img.size[0] * SCALE, img.size[1] * SCALE), (0, 0, 0, 0))
    video_frames = []

    # Process each channel
    for channel_idx, channel_img in enumerate(cmyk_channels):
        channel_name = ["Cyan", "Magenta", "Yellow", "Black"][channel_idx]
        root.title(f"Processing channel {channel_name}...")
        channel_img = ImageOps.grayscale(Image.fromarray(channel_img))
        channel_img = np.array(channel_img)
        with contextlib.redirect_stdout(StdoutRedirector(output_text)):
            pin_sequence, result, line_number, current_absdiff, frames = string_art(
                N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, channel_img
            )
        results.append(np.array(result))  # Ensure result is a numpy array
        total_lines.append(line_number)
        diffs.append(current_absdiff)
        frame_data.append(frames)

        if SAVE_MP4:
          idx_color = [(100, 0, 0, 0), (0, 100, 0, 0), (0, 0, 100, 0), (0, 0, 0, 100)][channel_idx]

          def reconstruct_frame(lines, frame):
              draw_frame = trasparent_frame.copy()
              draw = ImageDraw.Draw(draw_frame)
              draw.line(lines, fill=idx_color, width=1)
              if channel_name == "Black":
                  frame = ImageChops.subtract(frame, draw_frame)
              else:
                  frame = ImageChops.add(frame, draw_frame)
              del draw
              return frame

          pbar_label.config(text=f"{channel_name} channel reconstruction...")
          progress_bar["value"] = 0
          tk_pbar = progress_bar["value"]
          progress_bar["maximum"] = len(frames)
          root.update_idletasks()
          loop_ips = 1

          with tqdm(total=(len(frames))) as pbar:
              for frame_idx, frame_data in enumerate(frames):
                  video_frame = reconstruct_frame(frame_data, frame)
                  frame = video_frame
                  resized_frame = video_frame.resize((512, 512), Image.Resampling.BOX).convert("RGB")
                  if not INVERT:
                      resized_frame = ImageOps.invert(resized_frame)
                  video_frames.append(resized_frame)

                  # stats update
                  tk_pbar = tk_pbar + 1
                  progress_bar["value"] = tk_pbar
                  pbar_dict = pbar.format_dict
                  loop_time = round(pbar_dict["elapsed"])
                  if frame_idx % 10 == 0:
                      loop_ips = pbar_dict["rate"]
                      if loop_ips is not None:
                          loop_ips = round(loop_ips * 100)
                          loop_ips = loop_ips / 100
                      else:
                          loop_ips = 1

                  start_eta = len(frames) / loop_ips
                  current_eta = start_eta - loop_time
                  current_minutes, current_seconds = find_time(current_eta)

                  eta_label.config(
                      text=f"ETA: {current_minutes:02}:{current_seconds:02} | FPS: {loop_ips} | Frame: {frame_idx + 1}/{len(frames)}"
                  )
                  pbar.update(1)
                  root.update_idletasks()

    if SAVE_MP4:
    # Save the frames as an MP4 video
      clip = ImageSequenceClip(
          [np.array(frame) for frame in video_frames], fps=((sum(total_lines) / 17) / 4)
      )
      with contextlib.redirect_stdout(StdoutRedirector(output_text)):
        clip.write_videofile((output_path + "CMYK_output.mp4"), codec="libx264")

    if SAVE_JSON:
      with open((output_path + "_CMKY.json"), "w") as f:
        f.write(str(pin_sequence))

    root.title("Edit Settings")

    # Ensure we have exactly four channels
    if len(results) != 4:
        raise ValueError("Expected 4 channels for CMYK image, got {}".format(len(results)))

    # Merge the results into a single CMYK image
    combined_result = Image.merge("CMYK", [Image.fromarray(channel) for channel in results])

    return combined_result, total_lines, diffs


def sync_min_loop_entry(event):
    min_loop_entry.delete(0, tk.END)
    min_loop_entry.insert(0, min_loop_slider.get())


def sync_min_loop_slider(event):
    min_loop_slider.set(min_loop_entry.get())


def sync_min_distance_entry(event):
    min_distance_entry.delete(0, tk.END)
    min_distance_entry.insert(0, min_distance_slider.get())


def sync_min_distance_slider(event):
    min_distance_slider.set(min_distance_entry.get())


def sync_scale_entry(event):
    scale_entry.delete(0, tk.END)
    scale_entry.insert(0, scale_slider.get())


def sync_scale_slider(event):
    scale_slider.set(scale_entry.get())


def sync_line_weight_entry(event):
    line_weight_entry.delete(0, tk.END)
    line_weight_entry.insert(0, line_weight_slider.get())


def sync_line_weight_slider(event):
    line_weight_slider.set(line_weight_entry.get())


def update_label_text(event):
    pbar_label.config(text=root.title())

# Tkinter window colors and theme
TK_BG = "#272727"
TK_FG = "#d1d1d1"
TK_SEL_BG = "#464646"
padding_options = {"padx": 10, "pady": 5, "sticky": "w"}


root.configure(bg=TK_BG)

set_lines_label = tk.Label(root, text="SET_LINES", bg=TK_BG, fg=TK_FG)
set_lines_label.grid(row=0, column=0, **padding_options)

set_lines_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG, width=6)
set_lines_entry.insert(0, SET_LINES)
set_lines_entry.grid(row=0, column=1, **padding_options)

n_pins_label = tk.Label(root, text="N_PINS", bg=TK_BG, fg=TK_FG)
n_pins_label.grid(row=0, column=2, **padding_options)

n_pins_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG, width=6)
n_pins_entry.insert(0, N_PINS)
n_pins_entry.grid(row=0, column=3, **padding_options)

min_loop_label = tk.Label(root, text="MIN_LOOP", bg=TK_BG, fg=TK_FG)
min_loop_label.grid(row=2, column=0, **padding_options)

min_loop_slider = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, bg=TK_SEL_BG, fg=TK_FG)
min_loop_slider.set(MIN_LOOP)
min_loop_slider.grid(row=2, column=2)
min_loop_slider.bind("<Motion>", sync_min_loop_entry)

min_loop_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG, width=3)
min_loop_entry.insert(0, MIN_LOOP)
min_loop_entry.grid(row=2, column=1, **padding_options)
min_loop_entry.bind("<KeyRelease>", sync_min_loop_slider)

min_distance_label = tk.Label(root, text="MIN_DISTANCE", bg=TK_BG, fg=TK_FG)
min_distance_label.grid(row=3, column=0, **padding_options)

min_distance_slider = tk.Scale(root, from_=1, to=50, orient=tk.HORIZONTAL, bg=TK_SEL_BG, fg=TK_FG)
min_distance_slider.set(MIN_DISTANCE)
min_distance_slider.grid(row=3, column=2)
min_distance_slider.bind("<Motion>", sync_min_distance_entry)

min_distance_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG, width=3)
min_distance_entry.insert(0, MIN_DISTANCE)
min_distance_entry.grid(row=3, column=1, **padding_options)
min_distance_entry.bind("<KeyRelease>", sync_min_distance_slider)

line_weight_label = tk.Label(root, text="LINE_WEIGHT", bg=TK_BG, fg=TK_FG)
line_weight_label.grid(row=4, column=0, **padding_options)

line_weight_slider = tk.Scale(root, from_=1, to=80, orient=tk.HORIZONTAL, bg=TK_SEL_BG, fg=TK_FG)
line_weight_slider.set(LINE_WEIGHT)
line_weight_slider.grid(row=4, column=2)
line_weight_slider.bind("<Motion>", sync_line_weight_entry)

line_weight_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG, width=3)
line_weight_entry.insert(0, LINE_WEIGHT)
line_weight_entry.grid(row=4, column=1, **padding_options)
line_weight_entry.bind("<KeyRelease>", sync_line_weight_slider)

scale_label = tk.Label(root, text="SCALE", bg=TK_BG, fg=TK_FG)
scale_label.grid(row=5, column=0, **padding_options)

scale_slider = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL, bg=TK_SEL_BG, fg=TK_FG)
scale_slider.set(SCALE)
scale_slider.grid(row=5, column=2)
scale_slider.bind("<Motion>", sync_scale_entry)

scale_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG, width=3)
scale_entry.insert(0, SCALE)
scale_entry.grid(row=5, column=1, **padding_options)
scale_entry.bind("<KeyRelease>", sync_scale_slider)

grayscale_label = tk.Label(root, text="GRAYSCALE", bg=TK_BG, fg=TK_FG)
grayscale_label.grid(row=6, column=0, **padding_options)

grayscale_check = tk.Checkbutton(root, variable=grayscale_var, bg=TK_BG, fg=TK_FG, selectcolor=TK_SEL_BG)
grayscale_check.grid(row=6, column=1, **padding_options)

invert_label = tk.Label(root, text="INVERT", bg=TK_BG, fg=TK_FG)
invert_label.grid(row=7, column=0, **padding_options)

invert_check = tk.Checkbutton(root, variable=invert_var, bg=TK_BG, fg=TK_FG, selectcolor=TK_SEL_BG)
invert_check.grid(row=7, column=1, **padding_options)

save_mp4_label = tk.Label(root, text="SAVE_MP4", bg=TK_BG, fg=TK_FG)
save_mp4_label.grid(row=6, column=2, **padding_options)

save_mp4_check = tk.Checkbutton(root, variable=mp4_var, bg=TK_BG, fg=TK_FG, selectcolor=TK_SEL_BG)
save_mp4_check.grid(row=6, column=3, **padding_options)

save_json_label = tk.Label(root, text="SAVE_JSON", bg=TK_BG, fg=TK_FG)
save_json_label.grid(row=7, column=2, **padding_options)

save_json_check = tk.Checkbutton(root,variable=json_var, bg=TK_BG, fg=TK_FG, selectcolor=TK_SEL_BG)
save_json_check.grid(row=7, column=3, **padding_options)

file_path_label = tk.Label(root, text="FILE_PATH", bg=TK_BG, fg=TK_FG)
file_path_label.grid(row=8, column=0, **padding_options)

file_path_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG)
file_path_entry.insert(0, FILE_PATH)
file_path_entry.grid(row=8, column=1, columnspan=2, **padding_options)

tk.Button(root, text="Browse", command=select_file, bg=TK_SEL_BG, fg=TK_FG).grid(row=8, columnspan=99)

tk.Button(root, text="Run Code", command=run_code, bg=TK_SEL_BG, fg=TK_FG).grid(row=9, columnspan=99)

# display the image
image_label = tk.Label(root)
image_label.grid(row=11, columnspan=99, padx=10, pady=5)

# Text widget to display terminal prints
output_text = tk.Text(root, bg=TK_SEL_BG, fg=TK_FG, wrap="word", height=3, width=70)
output_text.grid(row=12, columnspan=99, padx=10, pady=5)

# progress bar
pbar_label = tk.Label(root, bg=TK_BG, fg=TK_FG)
pbar_label.grid(row=13, columnspan=99, padx=10, pady=5)
progress_bar = ttk.Progressbar(root, mode="determinate", length="135m")
progress_bar.grid(row=14, columnspan=99, padx=10, pady=5)
eta_label = tk.Label(root, bg=TK_BG, fg=TK_FG)
eta_label.grid(row=15, columnspan=99, padx=10, pady=5)

# tooltips
set_lines_tip = ToolTip(set_lines_label, "Specify the number of lines to draw. Set to 0 for automatic calculation.")
n_pins_tip = ToolTip(n_pins_label, "Set the total number of pins to use. must be a multiple of 36.")
min_loop_tip = ToolTip(min_loop_label, "Define the minimum loop count before returning to the same pin.")
min_distance_tip = ToolTip(min_distance_label, "Set the minimum distance between two pins.")
line_weight_tip = ToolTip(line_weight_label, "Adjust the weight of lines in error calculations. Higher values result in denser line packing.")
scale_tip = ToolTip(scale_label, "Set the scale factor for line calculations. Higher values improve accuracy but slow down processing.")
grayscale_tip = ToolTip(grayscale_label, "Convert the image to grayscale, using only black lines for drawing.")
invert_tip = ToolTip(invert_label, "Invert the image before processing. Can improve results for color images.")
save_mp4_tip = ToolTip(save_mp4_label, "Save the creation process as an MP4 video file.")
save_json_tip = ToolTip(save_json_label, "Save the pin sequence in a JSON file.")


root.mainloop()
