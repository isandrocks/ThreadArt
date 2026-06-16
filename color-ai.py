import math
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.draw import line_nd
from skimage.filters import sobel


COLOR_ORDER = ("C", "M", "Y", "K")
THREAD_RGB = {
    "C": (0, 255, 255),
    "M": (255, 0, 255),
    "Y": (255, 255, 0),
    "K": (0, 0, 0),
}


def create_circle_mask(resolution):
    yy, xx = np.ogrid[:resolution, :resolution]
    center = (resolution - 1) / 2.0
    radius = center - 2.0
    return ((yy - center) ** 2 + (xx - center) ** 2) <= radius**2


def get_pin_coordinates(num_pins, radius, center):
    pins = []
    for i in range(num_pins):
        angle = 2 * math.pi * i / num_pins
        row = int(round(center[0] + radius * math.sin(angle)))
        col = int(round(center[1] + radius * math.cos(angle)))
        pins.append((row, col))
    return pins


def prepare_square_image(uploaded_file, resolution):
    img = Image.open(uploaded_file).convert("RGB")
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    return img.resize((resolution, resolution), Image.Resampling.LANCZOS)


def rgb_to_thread_targets(img_arr):
    img_float = img_arr.astype(np.float32)
    r = img_float[:, :, 0]
    g = img_float[:, :, 1]
    b = img_float[:, :, 2]

    common_black = np.minimum(np.minimum(255.0 - r, 255.0 - g), 255.0 - b)
    c_target = (255.0 - r) - common_black
    m_target = (255.0 - g) - common_black
    y_target = (255.0 - b) - common_black
    k_target = common_black

    return {
        "C": np.clip(c_target, 0.0, 255.0),
        "M": np.clip(m_target, 0.0, 255.0),
        "Y": np.clip(y_target, 0.0, 255.0),
        "K": np.clip(k_target, 0.0, 255.0),
    }


def apply_mask_to_targets(targets, mask):
    masked_targets = {}
    for color in COLOR_ORDER:
        channel = targets[color].copy()
        channel[~mask] = 0.0
        masked_targets[color] = channel
    return masked_targets


def build_line_quota(total_lines):
    quota = {color: 0 for color in COLOR_ORDER}
    for i in range(total_lines):
        quota[COLOR_ORDER[i % len(COLOR_ORDER)]] += 1
    return quota


def build_dynamic_budget(targets, total_lines):
    energy = {color: float(targets[color].sum()) for color in COLOR_ORDER}
    total_energy = sum(energy.values())

    if total_energy <= 0:
        return build_line_quota(total_lines)

    raw_budget = {color: (energy[color] / total_energy) * total_lines for color in COLOR_ORDER}
    budget = {color: int(math.floor(raw_budget[color])) for color in COLOR_ORDER}

    assigned = sum(budget.values())
    remaining_slots = total_lines - assigned
    remainders = sorted(
        COLOR_ORDER,
        key=lambda color: (raw_budget[color] - budget[color], -COLOR_ORDER.index(color)),
        reverse=True,
    )
    for idx in range(remaining_slots):
        budget[remainders[idx % len(remainders)]] += 1

    # Pastikan warna yang memang punya kontribusi tetap bisa dipilih minimal 1 kali.
    active_colors = [color for color in COLOR_ORDER if energy[color] > 0]
    if active_colors and total_lines >= len(active_colors):
        for color in active_colors:
            if budget[color] == 0:
                donor = max(COLOR_ORDER, key=lambda item: budget[item])
                if budget[donor] > 1:
                    budget[donor] -= 1
                    budget[color] += 1

    return budget


def precalculate_lines(pins, resolution, min_pin_gap):
    num_pins = len(pins)
    line_cache = [None] * (num_pins * num_pins)

    for pin_a in range(num_pins):
        for pin_b in range(pin_a + min_pin_gap, num_pins):
            rr, cc = line_nd(pins[pin_a], pins[pin_b], endpoint=True)
            rr = np.clip(rr.astype(np.int16), 0, resolution - 1)
            cc = np.clip(cc.astype(np.int16), 0, resolution - 1)
            line_cache[pin_a * num_pins + pin_b] = (rr, cc)
            line_cache[pin_b * num_pins + pin_a] = (rr, cc)

    return line_cache


def pick_best_pin(
    current_pin,
    num_pins,
    error_map,
    edge_map,
    min_pin_gap,
    candidate_stride,
    recent_pins,
    line_cache,
    line_strength,
):
    best_pin = -1
    best_score = -math.inf
    best_coords = None

    for offset in range(min_pin_gap, num_pins - min_pin_gap, candidate_stride):
        test_pin = (current_pin + offset) % num_pins
        if test_pin in recent_pins:
            continue

        coords = line_cache[current_pin * num_pins + test_pin]
        if coords is None:
            continue
        rr, cc = coords

        error_values = error_map[rr, cc]
        if error_values.size == 0:
            continue

        edge_values = edge_map[rr, cc]
        positive = np.minimum(error_values, line_strength) * (1.0 + edge_values)
        score = float(positive.sum())

        if score > best_score:
            best_score = score
            best_pin = test_pin
            best_coords = (rr, cc)

    return best_pin, best_score, best_coords


def render_channel_preview(channel_map):
    return Image.fromarray(np.clip(255.0 - channel_map, 0, 255).astype(np.uint8), mode="L")


def composite_cmyk_to_rgb(coverage, mask):
    c = np.clip(coverage["C"], 0.0, 255.0) / 255.0
    m = np.clip(coverage["M"], 0.0, 255.0) / 255.0
    y = np.clip(coverage["Y"], 0.0, 255.0) / 255.0
    k = np.clip(coverage["K"], 0.0, 255.0) / 255.0

    r = 255.0 * (1.0 - c) * (1.0 - k)
    g = 255.0 * (1.0 - m) * (1.0 - k)
    b = 255.0 * (1.0 - y) * (1.0 - k)

    rgb = np.stack([r, g, b], axis=-1)
    rgb[~mask] = 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def generate_string_art(
    targets,
    pins,
    resolution,
    total_lines,
    min_pin_gap,
    candidate_stride,
    thread_thickness,
    preview_every,
    edge_boost,
    progress_bar,
    status_text,
    image_placeholder,
):
    num_pins = len(pins)
    current_pins = {color: 0 for color in COLOR_ORDER}
    recent_pins = {color: deque(maxlen=10) for color in COLOR_ORDER}
    master_sequence = []
    circle_mask = create_circle_mask(resolution)
    line_cache = precalculate_lines(pins, resolution, min_pin_gap)

    quota = build_dynamic_budget(targets, total_lines)
    remaining = quota.copy()
    exhausted = set()

    strength_scale = max(thread_thickness / 0.2, 0.5)
    line_strength = 18.0 * strength_scale
    error_maps = {}
    edge_maps = {}
    coverage = {}
    for color in COLOR_ORDER:
        coverage[color] = np.zeros((resolution, resolution), dtype=np.float32)
        error_maps[color] = targets[color].astype(np.float32).copy()
        base_edge = sobel(targets[color] / 255.0).astype(np.float32)
        if float(base_edge.max()) > 0.0:
            base_edge /= float(base_edge.max())
        edge_maps[color] = np.clip(base_edge * edge_boost, 0.0, 3.0)

    step = 0
    while step < total_lines:
        best_choice = None

        for color in COLOR_ORDER:
            if remaining[color] <= 0 or color in exhausted:
                continue

            best_pin, best_score, best_coords = pick_best_pin(
                current_pins[color],
                num_pins,
                error_maps[color],
                edge_maps[color],
                min_pin_gap,
                candidate_stride,
                recent_pins[color],
                line_cache,
                line_strength,
            )

            if best_pin == -1 or best_coords is None or best_score <= 0.01:
                exhausted.add(color)
                continue

            choice = {
                "color": color,
                "pin": best_pin,
                "score": best_score,
                "coords": best_coords,
            }
            if best_choice is None or choice["score"] > best_choice["score"]:
                best_choice = choice

        if best_choice is None:
            break

        color = best_choice["color"]
        best_pin = best_choice["pin"]
        rr, cc = best_choice["coords"]
        start_pin = current_pins[color]

        coverage[color][rr, cc] = np.clip(coverage[color][rr, cc] + line_strength, 0.0, 255.0)
        error_maps[color][rr, cc] -= line_strength
        np.clip(error_maps[color], 0.0, 255.0, out=error_maps[color])

        master_sequence.append(
            {
                "STEP": step + 1,
                "COLOR": color,
                "START_PIN": start_pin,
                "END_PIN": best_pin,
                "SCORE": round(best_choice["score"], 2),
            }
        )

        current_pins[color] = best_pin
        recent_pins[color].append(start_pin)
        recent_pins[color].append(best_pin)
        remaining[color] -= 1
        step += 1

        if step % preview_every == 0 or step == total_lines:
            composite_preview = composite_cmyk_to_rgb(coverage, circle_mask)
            progress_bar.progress(step / total_lines)
            status_text.text(
                f"Generating {step}/{total_lines} lines | "
                f"C:{quota['C'] - remaining['C']} M:{quota['M'] - remaining['M']} "
                f"Y:{quota['Y'] - remaining['Y']} K:{quota['K'] - remaining['K']}"
            )
            image_placeholder.image(
                composite_preview,
                caption="Preview komposit CMYK",
                use_container_width=True,
            )

    progress_bar.progress(1.0 if total_lines else 0.0)
    final_canvas = composite_cmyk_to_rgb(coverage, circle_mask)
    return master_sequence, final_canvas, coverage, quota, remaining


st.set_page_config(layout="wide", page_title="CMYK String Art Generator")
st.title("CMYK String Art Generator")
st.caption("Generator string art CMYK dengan perpindahan warna dinamis sesuai kebutuhan gambar.")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Konfigurasi")
    uploaded_file = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg"])
    thread_thickness = st.number_input("Ketebalan benang (mm)", min_value=0.1, max_value=2.0, value=0.2, step=0.1)
    num_pins = st.number_input("Jumlah paku", min_value=60, max_value=360, value=180, step=10)
    num_lines = st.number_input("Total tarikan benang", min_value=100, max_value=12000, value=1200, step=100)
    resolution = st.number_input("Resolusi kerja", min_value=200, max_value=900, value=420, step=20)
    min_pin_gap = st.number_input("Jarak minimum antar paku", min_value=4, max_value=80, value=12, step=1)
    candidate_stride = st.number_input("Langkah pencarian paku", min_value=1, max_value=12, value=2, step=1)
    edge_boost = st.slider("Prioritas detail tepi", min_value=0.5, max_value=3.0, value=1.6, step=0.1)
    preview_every = st.number_input("Update preview tiap berapa garis", min_value=5, max_value=200, value=20, step=5)
    generate_btn = st.button("Generate CSV")

with right_col:
    if uploaded_file is not None:
        preview_img = prepare_square_image(uploaded_file, 320)
        st.image(preview_img, caption="Preview gambar upload", use_container_width=True)

if uploaded_file is not None and generate_btn:
    with right_col:
        st.subheader("Proses")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        image_placeholder = st.empty()

        prepared_img = prepare_square_image(uploaded_file, int(resolution))
        img_arr = np.array(prepared_img)
        circle_mask = create_circle_mask(int(resolution))
        targets = apply_mask_to_targets(rgb_to_thread_targets(img_arr), circle_mask)
        pins = get_pin_coordinates(int(num_pins), (int(resolution) // 2) - 10, (int(resolution) // 2, int(resolution) // 2))

        master_sequence, final_canvas, coverage, quota, remaining = generate_string_art(
            targets=targets,
            pins=pins,
            resolution=int(resolution),
            total_lines=int(num_lines),
            min_pin_gap=int(min_pin_gap),
            candidate_stride=int(candidate_stride),
            thread_thickness=float(thread_thickness),
            preview_every=int(preview_every),
            edge_boost=float(edge_boost),
            progress_bar=progress_bar,
            status_text=status_text,
            image_placeholder=image_placeholder,
        )

        df_sequence = pd.DataFrame(master_sequence)
        csv_data = df_sequence.to_csv(index=False)

        used_lines = len(master_sequence)
        st.success(
            f"Selesai. Terpakai {used_lines} garis dari target {int(num_lines)}."
        )
        if used_lines < int(num_lines):
            st.warning(
                "Sebagian kuota berhenti lebih awal karena area warna terkait sudah hampir habis. "
                "Coba naikkan resolusi, jumlah paku, atau kurangi jarak minimum."
            )

        summary_df = pd.DataFrame(
            [
                {
                    "COLOR": color,
                    "TARGET_LINES": quota[color],
                    "USED_LINES": quota[color] - remaining[color],
                }
                for color in COLOR_ORDER
            ]
        )

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.image(prepared_img, caption="Gambar target", use_container_width=True)
        with result_col2:
            st.image(final_canvas, caption="Simulasi hasil string art", use_container_width=True)

        st.subheader("Distribusi Warna")
        st.dataframe(summary_df, use_container_width=True)

        channel_cols = st.columns(4)
        for idx, color in enumerate(COLOR_ORDER):
            with channel_cols[idx]:
                st.image(
                    render_channel_preview(coverage[color]),
                    caption=f"Coverage {color}",
                    use_container_width=True,
                )

        st.subheader("Download Instruksi Robot")
        st.download_button(
            label="Download robot_sequence.csv",
            data=csv_data,
            file_name="robot_sequence.csv",
            mime="text/csv",
        )

        st.subheader("Preview CSV")
        st.dataframe(df_sequence.head(20), use_container_width=True)
