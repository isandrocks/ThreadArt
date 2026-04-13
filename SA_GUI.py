from PIL import Image, ImageOps, ImageTk, ImageDraw, ImageChops
import numpy as np
from SAapp import string_art, string_art_multiscale
from SA_DQN import string_art_dqn, string_art_dqn_multiscale
import os
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import queue
from moviepy import ImageSequenceClip
import contextlib
from tqdm import tqdm
from skimage.filters import sobel

# Default settings
SET_LINES = 0
N_PINS = 36 * 8
MIN_LOOP = 5
MIN_DISTANCE = 5
LINE_WEIGHT = 25
SCALE = 4
INVERT = False
FILE_PATH = ""
GRAYSCALE = True
SAVE_MP4 = False
SAVE_CSV = False
DQN_MODE = False
TRAINING_EPISODES = 15
AUTO_LW = True
SSIM_TARGET = 0.65
PREPROCESS = True
MULTI_SCALE = False

# Tkinter root window
root = tk.Tk()
root.title("Edit Settings")
root.geometry("+0+0")

invert_var = tk.BooleanVar(value=INVERT)
grayscale_var = tk.BooleanVar(value=GRAYSCALE)
mp4_var = tk.BooleanVar(value=SAVE_MP4)
csv_var = tk.BooleanVar(value=SAVE_CSV)
dqn_var = tk.BooleanVar(value=DQN_MODE)
auto_lw_var = tk.BooleanVar(value=AUTO_LW)
preprocess_var = tk.BooleanVar(value=PREPROCESS)
multi_scale_var = tk.BooleanVar(value=MULTI_SCALE)

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


def auto_line_weight(img_array, scale=4):
    """Compute LINE_WEIGHT from image statistics and render scale.

    Targets ~4 passes over an average-error pixel to fully resolve it.
    Darker images get higher weight, lighter images get lower weight.
    """
    length = img_array.shape[0]
    X, Y = np.ogrid[0:length, 0:length]
    circle = (X - length / 2) ** 2 + (Y - length / 2) ** 2 <= (length / 2) ** 2
    error = 255.0 - img_array[circle].astype(np.float64)
    # Only consider pixels with meaningful error
    significant = error[error > 10]
    mean_err = float(significant.mean()) if len(significant) > 0 else float(error.mean())
    std_err = float(significant.std()) if len(significant) > 0 else float(error.std())
    # Contrast ratio: high std relative to mean = high contrast image.
    # Low-contrast images need lower weight to preserve subtle tonal differences.
    contrast_ratio = std_err / max(mean_err, 1.0)
    contrast_factor = 0.7 + 0.6 * min(contrast_ratio, 0.5)
    # Base weight from image content (6-pass target for finer detail)
    base = mean_err / 6 * contrast_factor
    weight = int(np.clip(base, 10, 50))
    return weight


def update_settings():
    global SET_LINES, N_PINS, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, INVERT, FILE_PATH, GRAYSCALE, SAVE_MP4, SAVE_CSV, DQN_MODE, TRAINING_EPISODES, AUTO_LW
    global SSIM_TARGET, PREPROCESS, MULTI_SCALE
    SET_LINES = int(set_lines_entry.get())
    N_PINS = int(n_pins_slider.get())
    MIN_LOOP = int(min_loop_slider.get())
    MIN_DISTANCE = int(min_distance_slider.get())
    LINE_WEIGHT = int(line_weight_slider.get())
    SCALE = int(scale_slider.get())
    INVERT = invert_var.get()
    FILE_PATH = file_path_entry.get()
    GRAYSCALE = grayscale_var.get()
    SAVE_MP4 = mp4_var.get()
    SAVE_CSV = csv_var.get()
    DQN_MODE = dqn_var.get()
    TRAINING_EPISODES = int(training_episodes_entry.get())
    AUTO_LW = auto_lw_var.get()
    SSIM_TARGET = float(ssim_target_slider.get())
    PREPROCESS = preprocess_var.get()
    MULTI_SCALE = multi_scale_var.get()


def run_code():
    if FILE_PATH == "":
        with contextlib.redirect_stdout(StdoutRedirector(output_text)):
            print("Please select an image file.")
        return
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

    # Stretch the histogram to use the full tonal range for better contrast
    img = ImageOps.autocontrast(img, cutoff=1)

    if PREPROCESS:
        from skimage.restoration import denoise_bilateral
        img_np = np.array(img)
        filtered = denoise_bilateral(img_np, sigma_spatial=3, sigma_color=0.1, channel_axis=-1)
        bins = np.linspace(0, 1, 6)
        digitized = np.digitize(filtered, bins) - 1
        posterized = (digitized * (255.0 / 4.0)).astype(np.uint8)
        img = Image.fromarray(posterized)

    # Auto-compute LINE_WEIGHT from image content
    lw = LINE_WEIGHT
    if AUTO_LW:
        gray_for_lw = np.array(ImageOps.grayscale(img))
        lw = auto_line_weight(gray_for_lw, scale=SCALE)
        with contextlib.redirect_stdout(StdoutRedirector(output_text)):
            print(f"Auto LINE_WEIGHT: {lw}")

    if GRAYSCALE:
        result, length, current_absdiff = string_art_grayscale(
            N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, lw, SCALE, img
        )
        length = [length]  # Ensure length is a list
    else:
        result, length, current_absdiff = string_art_cmyk(
            N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, lw, SCALE, img
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
    
    edge_map = sobel(gray_channel.astype(np.float64))
    if edge_map.max() > 0:
        edge_map /= edge_map.max()

    with contextlib.redirect_stdout(StdoutRedirector(output_text)):
        if MULTI_SCALE:
            if DQN_MODE:
                print("Processing grayscale channel (DQN Multi-Scale)...")
                pin_sequence, result, line_number, current_absdiff, frames = string_art_dqn_multiscale(
                    N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, gray_channel,
                    edge_map=edge_map, SSIM_TARGET=SSIM_TARGET,
                    training_episodes=TRAINING_EPISODES,
                    no_stagnation=(SET_LINES != 0)
                )
            else:
                print("Processing grayscale channel (Multi-Scale)...")
                pin_sequence, result, line_number, current_absdiff, frames = string_art_multiscale(
                    N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, gray_channel,
                    edge_map=edge_map, SSIM_TARGET=SSIM_TARGET,
                    no_stagnation=(SET_LINES != 0)
                )
        else:
            if DQN_MODE:
                print("Processing grayscale channel (DQN mode)...")
                pin_sequence, result, line_number, current_absdiff, frames = string_art_dqn(
                    N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, gray_channel,
                    edge_map=edge_map, SSIM_TARGET=SSIM_TARGET,
                    training_episodes=TRAINING_EPISODES,
                    no_stagnation=(SET_LINES != 0)
                )
            else:
                print("Processing grayscale channel...")
                pin_sequence, result, line_number, current_absdiff, frames = string_art(
                    N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, gray_channel,
                    edge_map=edge_map, SSIM_TARGET=SSIM_TARGET,
                    no_stagnation=(SET_LINES != 0)
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
        clip = ImageSequenceClip([np.array(frame) for frame in video_frames], fps=(line_number / 17))
        with contextlib.redirect_stdout(StdoutRedirector(output_text)):
            clip.write_videofile((output_path + "_grayscale_output.mp4"), codec="libx264")

    if SAVE_CSV:
        import csv
        with open((output_path + ".csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pin"])
            for pin in pin_sequence:
                writer.writerow([pin])

    return result_img, line_number, current_absdiff


def string_art_cmyk(N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img):
    # Convert image to CMYK
    img_cmyk = img.convert("CMYK")

    cmyk_channels = [np.array(img_cmyk.getchannel(channel)) for channel in range(4)]

    results = []  # To store results for each channel
    total_lines = []  # To store line counts for each channel
    diffs = []  # To store errors for each channel
    frame_data = []  # To store frame data for each channel
    frame_np = np.zeros((img.size[1] * SCALE, img.size[0] * SCALE, 4), dtype=np.int16)
    video_frames = []
    SNAPSHOT_EVERY = 4  # only snapshot every Nth line for the video

    # Process each channel
    for channel_idx, channel_img in enumerate(cmyk_channels):
        channel_name = ["Cyan", "Magenta", "Yellow", "Black"][channel_idx]
        root.title(f"Processing channel {channel_name}...")
        pbar_label.config(text=f"Processing channel {channel_name}...")
        channel_img = ImageOps.grayscale(Image.fromarray(channel_img))
        channel_img = np.array(channel_img)
        if channel_name == "Black":
            gs_img = img_cmyk.convert("L")
            gs_img = ImageOps.invert(gs_img)
            channel_img = np.array(gs_img)
        
        edge_map = sobel(channel_img.astype(np.float64))
        if edge_map.max() > 0:
            edge_map /= edge_map.max()

        # Per-channel auto LINE_WEIGHT
        channel_lw = LINE_WEIGHT
        if AUTO_LW:
            channel_lw = auto_line_weight(channel_img, scale=SCALE)
            print(f"  Auto LINE_WEIGHT for {channel_name}: {channel_lw}")
        with contextlib.redirect_stdout(StdoutRedirector(output_text)):
            if MULTI_SCALE:
                if DQN_MODE:
                    pin_sequence, result, line_number, current_absdiff, frames = string_art_dqn_multiscale(
                        N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, channel_lw, SCALE, channel_img,
                        edge_map=edge_map, SSIM_TARGET=SSIM_TARGET,
                        training_episodes=TRAINING_EPISODES,
                        no_stagnation=(SET_LINES != 0)
                    )
                else:
                    pin_sequence, result, line_number, current_absdiff, frames = string_art_multiscale(
                        N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, channel_lw, SCALE, channel_img,
                        edge_map=edge_map, SSIM_TARGET=SSIM_TARGET,
                        no_stagnation=(SET_LINES != 0)
                    )
            else:
                if DQN_MODE:
                    pin_sequence, result, line_number, current_absdiff, frames = string_art_dqn(
                        N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, channel_lw, SCALE, channel_img,
                        edge_map=edge_map, SSIM_TARGET=SSIM_TARGET,
                        training_episodes=TRAINING_EPISODES,
                        no_stagnation=(SET_LINES != 0)
                    )
                else:
                    pin_sequence, result, line_number, current_absdiff, frames = string_art(
                        N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, channel_lw, SCALE, channel_img,
                        edge_map=edge_map, SSIM_TARGET=SSIM_TARGET,
                        no_stagnation=(SET_LINES != 0)
                    )
        results.append(np.array(result))  # Ensure result is a numpy array
        total_lines.append(line_number)
        diffs.append(current_absdiff)
        frame_data.append(frames)

        if SAVE_MP4:
            ink = np.array([(120, 0, 0, 0), (0, 120, 0, 0), (0, 0, 120, 0), (0, 0, 0, 120)][channel_idx], dtype=np.int16)
            subtract = channel_name == "Black"
            frame_h, frame_w = frame_np.shape[:2]

            pbar_label.config(text=f"{channel_name} channel reconstruction...")
            progress_bar["value"] = 0
            tk_pbar = 0
            progress_bar["maximum"] = len(frames)
            root.update_idletasks()
            loop_ips = 1

            with tqdm(total=len(frames)) as pbar:
                for frame_idx, line_data in enumerate(frames):
                    # Draw line directly into numpy accumulator
                    (x0, y0), (x1, y1) = line_data
                    d = max(abs(x1 - x0), abs(y1 - y0), 1)
                    xs = np.linspace(x0, x1, d, dtype=int)
                    ys = np.linspace(y0, y1, d, dtype=int)
                    valid = (xs >= 0) & (xs < frame_w) & (ys >= 0) & (ys < frame_h)
                    if subtract:
                        frame_np[ys[valid], xs[valid]] -= ink
                    else:
                        frame_np[ys[valid], xs[valid]] += ink

                    # Only snapshot every Nth frame for the video
                    if frame_idx % SNAPSHOT_EVERY == 0 or frame_idx == len(frames) - 1:
                        clipped = np.clip(frame_np, 0, 255).astype(np.uint8)
                        pil_frame = Image.fromarray(clipped, mode="CMYK")
                        resized_frame = pil_frame.resize((512, 512), Image.Resampling.BOX).convert("RGB")
                        if not INVERT:
                            resized_frame = ImageOps.invert(resized_frame)
                        video_frames.append(resized_frame)

                    tk_pbar += 1
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
        target_duration = 68  # ~17 seconds per channel
        clip_fps = max(1, len(video_frames) / target_duration)
        clip = ImageSequenceClip([np.array(f) for f in video_frames], fps=clip_fps)
        with contextlib.redirect_stdout(StdoutRedirector(output_text)):
            clip.write_videofile((output_path + f"_S_{SCALE}_LW_{LINE_WEIGHT}_CMYK_output.mp4"), codec="libx264")

    if SAVE_CSV:
        import csv
        with open((output_path + "_CMYK.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pin"])
            for pin in pin_sequence:
                writer.writerow([pin])

    root.title("Edit Settings")
    pbar_label.config(text=f"Completed processing {os.path.basename(FILE_PATH)}")

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

n_pins_slider = tk.Scale(
    root, from_=36, to=360, resolution=36, orient=tk.HORIZONTAL,
    bg=TK_SEL_BG, fg=TK_FG, length=150,
)
n_pins_slider.set(N_PINS)
n_pins_slider.grid(row=0, column=3, **padding_options)

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

auto_lw_label = tk.Label(root, text="AUTO_LW", bg=TK_BG, fg=TK_FG)
auto_lw_label.grid(row=4, column=3, **padding_options)

auto_lw_check = tk.Checkbutton(root, variable=auto_lw_var, bg=TK_BG, fg=TK_FG, selectcolor=TK_SEL_BG)
auto_lw_check.grid(row=4, column=4, **padding_options)

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

save_csv_label = tk.Label(root, text="SAVE_CSV", bg=TK_BG, fg=TK_FG)
save_csv_label.grid(row=7, column=2, **padding_options)

save_csv_check = tk.Checkbutton(root, variable=csv_var, bg=TK_BG, fg=TK_FG, selectcolor=TK_SEL_BG)
save_csv_check.grid(row=7, column=3, **padding_options)

dqn_mode_label = tk.Label(root, text="DQN_MODE", bg=TK_BG, fg=TK_FG)
dqn_mode_label.grid(row=8, column=0, **padding_options)

dqn_mode_check = tk.Checkbutton(root, variable=dqn_var, bg=TK_BG, fg=TK_FG, selectcolor=TK_SEL_BG)
dqn_mode_check.grid(row=8, column=1, **padding_options)

training_episodes_label = tk.Label(root, text="EPISODES", bg=TK_BG, fg=TK_FG)
training_episodes_label.grid(row=8, column=2, **padding_options)

training_episodes_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG, width=6)
training_episodes_entry.insert(0, TRAINING_EPISODES)
training_episodes_entry.grid(row=8, column=3, **padding_options)

ssim_target_label = tk.Label(root, text="SSIM_TARGET", bg=TK_BG, fg=TK_FG)
ssim_target_label.grid(row=9, column=0, **padding_options)

ssim_target_slider = tk.Scale(root, from_=0.5, to=0.9, resolution=0.01, orient=tk.HORIZONTAL, bg=TK_SEL_BG, fg=TK_FG)
ssim_target_slider.set(SSIM_TARGET)
ssim_target_slider.grid(row=9, column=2)

preprocess_label = tk.Label(root, text="PREPROCESS", bg=TK_BG, fg=TK_FG)
preprocess_label.grid(row=10, column=0, **padding_options)

preprocess_check = tk.Checkbutton(root, variable=preprocess_var, bg=TK_BG, fg=TK_FG, selectcolor=TK_SEL_BG)
preprocess_check.grid(row=10, column=1, **padding_options)

multi_scale_label = tk.Label(root, text="MULTI_SCALE", bg=TK_BG, fg=TK_FG)
multi_scale_label.grid(row=10, column=2, **padding_options)

multi_scale_check = tk.Checkbutton(root, variable=multi_scale_var, bg=TK_BG, fg=TK_FG, selectcolor=TK_SEL_BG)
multi_scale_check.grid(row=10, column=3, **padding_options)

file_path_label = tk.Label(root, text="FILE_PATH", bg=TK_BG, fg=TK_FG)
file_path_label.grid(row=11, column=0, **padding_options)

file_path_entry = tk.Entry(root, bg=TK_SEL_BG, fg=TK_FG)
file_path_entry.insert(0, FILE_PATH)
file_path_entry.grid(row=11, column=1, columnspan=2, **padding_options)

tk.Button(root, text="Browse", command=select_file, bg=TK_SEL_BG, fg=TK_FG).grid(row=11, column=3, **padding_options)

tk.Button(root, text="Run Code", command=run_code, bg=TK_SEL_BG, fg=TK_FG).grid(row=12, columnspan=99)

# display the image
image_label = tk.Label(root)
image_label.grid(row=13, columnspan=99, padx=10, pady=5)

# Text widget to display terminal prints
output_text = tk.Text(root, bg=TK_SEL_BG, fg=TK_FG, wrap="word", height=3, width=70)
output_text.grid(row=14, columnspan=99, padx=10, pady=5)

# progress bar
pbar_label = tk.Label(root, bg=TK_BG, fg=TK_FG)
pbar_label.grid(row=15, columnspan=99, padx=10, pady=5)
progress_bar = ttk.Progressbar(root, mode="determinate", length="135m")
progress_bar.grid(row=16, columnspan=99, padx=10, pady=5)
eta_label = tk.Label(root, bg=TK_BG, fg=TK_FG)
eta_label.grid(row=17, columnspan=99, padx=10, pady=5)

# tooltips
set_lines_tip = ToolTip(set_lines_label, "Set the number of lines to draw. Set to 0 for automatic calculation.")
n_pins_tip = ToolTip(n_pins_label, "Set the total number of pins to use (multiples of 36).")
min_loop_tip = ToolTip(min_loop_label, "Set the minimum loop count before returning to the same pin.")
min_distance_tip = ToolTip(min_distance_label, "Set the minimum distance between two pins.")
line_weight_tip = ToolTip(
    line_weight_label, "Set the weight of lines in error calculations. Higher values result in denser line packing."
)
scale_tip = ToolTip(
    scale_label, "Set the scale factor for line calculations. Higher values improve accuracy but slow down processing."
)
grayscale_tip = ToolTip(grayscale_label, "Convert the image to grayscale, using only black lines for drawing.")
invert_tip = ToolTip(invert_label, "Invert the image before processing. Can improve results for color images.")
save_mp4_tip = ToolTip(save_mp4_label, "Save the creation process as an MP4 video file.")
save_csv_tip = ToolTip(save_csv_label, "Save the pin sequence as a CSV file (one pin per row).")
dqn_mode_tip = ToolTip(dqn_mode_label, "Use a Deep Q-Network to learn pin placement instead of the greedy algorithm.")
training_episodes_tip = ToolTip(training_episodes_label, "Number of training episodes for the DQN (more = better quality, slower).")
auto_lw_tip = ToolTip(auto_lw_label, "Automatically compute LINE_WEIGHT from image darkness. Overrides the slider value.")
ssim_target_tip = ToolTip(ssim_target_label, "SSIM target for early stopping. Higher values result in more lines and higher quality.")
preprocess_tip = ToolTip(preprocess_label, "Apply bilateral filtering and posterization to the image before processing.")
multi_scale_tip = ToolTip(multi_scale_label, "Use a two-pass multiscale approach (coarse then fine) to save lines.")

root.mainloop()
