"""
═══════════════════════════════════════════════════════════════════════════════
 CMYK / K-Only String Art Generator v4
═══════════════════════════════════════════════════════════════════════════════

 Deskripsi:
   Aplikasi Streamlit untuk menghasilkan instruksi string art dari gambar.
   Mendukung mode CMYK (4 warna benang) atau K-only (hitam saja).
   Output berupa CSV berisi urutan pin & warna yang bisa dipakai robot
   atau sebagai panduan manual.

═══════════════════════════════════════════════════════════════════════════════
 SETUP & INSTALASI
═══════════════════════════════════════════════════════════════════════════════

 1. Python 3.9+ diperlukan

 2. Buat virtual environment (opsional tapi disarankan):
      python -m venv venv
      venv\\Scripts\\activate        (Windows)
      source venv/bin/activate      (Linux/Mac)

 3. Install dependencies:
      pip install -r requirements.txt

    Atau manual:
      pip install streamlit numpy pandas pillow scikit-image

 4. Jalankan:
      streamlit run color-ai.py

    Browser akan terbuka otomatis di http://localhost:8501

═══════════════════════════════════════════════════════════════════════════════
 CARA PAKAI
═══════════════════════════════════════════════════════════════════════════════

 1. Upload gambar (PNG/JPG) di sidebar kiri
 2. Sistem otomatis menganalisis gambar dan memberikan:
    - Preview dekomposisi CMYK
    - Recommended settings (mode, pins, lines, dll)
 3. Pilih mode warna:
    - CMYK = 4 warna benang (Cyan, Magenta, Yellow, Black)
    - K Only = hitam saja (untuk portrait/grayscale)
 4. Gunakan suggested settings (checkbox ON) atau edit manual (OFF)
 5. Klik "Generate" — progress bar & preview realtime
 6. Setelah selesai:
    - Lihat simulasi visual hasil
    - Cek tabel distribusi warna & estimasi kebutuhan benang
    - Download CSV instruksi

═══════════════════════════════════════════════════════════════════════════════
 PARAMETER GUIDE
═══════════════════════════════════════════════════════════════════════════════

 BASIC:
   - Jumlah paku        : Pin di keliling frame. 240-360 untuk kebanyakan gambar.
   - Total tarikan      : Jumlah garis. 4000-6000 biasa. Gambar gelap butuh lebih.
   - Resolusi kerja     : Resolusi kalkulasi internal. 400-550 biasanya optimal.
   - Edge boost         : Prioritas kontur/tepi. 1.5-2.5 range normal.
   - Highlight protect  : Pixel target < ini dianggap kosong. Naikkan untuk
                          protect area terang (kulit, putih). Range 12-30.
   - Diameter frame     : Pilih preset ukuran: 20, 30, 40, 50, 60 cm.
                          Mempengaruhi suggestion (pins, lines) dan kalkulasi benang.
   - Ketebalan benang   : Untuk kalkulasi line strength. Default 0.2mm.

 ADVANCED (Anti-Moiré):
   - Candidate stride   : Lompatan saat scan pin. 1=presisi, 3+=diverse.
                          K-only disarankan 3+.
   - Recent pins buffer : Pin terakhir yg di-blacklist. 15=CMYK, 40-60=K-only.
                          Kecil → pola repetitif. Besar → lebih diverse.
   - Min line length    : Tolak garis < X pixel. Besar = hilangkan sisik/moiré.
                          Default: resolution/8 (CMYK), resolution/5 (K-only).

═══════════════════════════════════════════════════════════════════════════════
 TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

 - Hasil ada "sisik"/moiré   → Naikkan min line length & recent buffer
 - Area terang kena garis    → Naikkan highlight protection (25-35)
 - Terlalu gelap/over-draw   → Kurangi total tarikan atau naikkan highlight
 - Proses freeze di awal     → Normal, precalculate garis butuh 10-30 detik
 - Hasil tipis/jarang        → Naikkan total tarikan, turunkan highlight
 - Garis kasar               → Naikkan jumlah paku atau resolusi

═══════════════════════════════════════════════════════════════════════════════
 OUTPUT CSV FORMAT
═══════════════════════════════════════════════════════════════════════════════

 Kolom CSV:
   STEP      : Nomor urut garis (1-based)
   COLOR     : Warna benang (C/M/Y/K)
   START_PIN : Nomor pin awal (0-based)
   END_PIN   : Nomor pin akhir (0-based)
   START_ROW : Koordinat baris pin awal (pixel)
   START_COL : Koordinat kolom pin awal (pixel)
   END_ROW   : Koordinat baris pin akhir (pixel)
   END_COL   : Koordinat kolom pin akhir (pixel)
   LENGTH_PX : Panjang garis dalam pixel
   LENGTH_MM : Panjang garis dalam mm (berdasarkan diameter frame)
   SCORE     : Skor kualitas garis (semakin tinggi = semakin efektif)

═══════════════════════════════════════════════════════════════════════════════
"""

import math
from collections import deque

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.draw import line_nd
from skimage.filters import sobel


COLOR_ORDER_CMYK = ("C", "M", "Y", "K")
COLOR_ORDER_K = ("K",)

THREAD_RGB = {
    "C": (0, 255, 255),
    "M": (255, 0, 255),
    "Y": (255, 255, 0),
    "K": (0, 0, 0),
}

CHANNEL_STRENGTH_MULTIPLIER = {
    "C": 1.0,
    "M": 1.0,
    "Y": 0.85,
    "K": 1.4,
}

HIGHLIGHT_THRESHOLD = 18.0


# ═══════════════════════════ UTILITY FUNCTIONS ═══════════════════════════


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


# ═══════════════════════════ IMAGE ANALYSIS ═══════════════════════════


def analyze_image(img_arr, circle_mask, frame_diameter_cm=50):
    """Analisis gambar dan berikan suggestion parameter.
    Return dict berisi analisis & recommended settings.
    """
    img_float = img_arr.astype(np.float32)
    r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    # Hitung dalam mask saja
    mask_pixels = circle_mask.sum()
    r_m = r[circle_mask]
    g_m = g[circle_mask]
    b_m = b[circle_mask]

    # Brightness
    luminance = 0.299 * r_m + 0.587 * g_m + 0.114 * b_m
    avg_brightness = float(luminance.mean())
    brightness_std = float(luminance.std())

    # Color saturation (HSV-style)
    max_rgb = np.maximum(np.maximum(r_m, g_m), b_m)
    min_rgb = np.minimum(np.minimum(r_m, g_m), b_m)
    chroma = max_rgb - min_rgb
    avg_saturation = float(chroma.mean())

    # Detail/complexity (edge density)
    gray = luminance.reshape(-1)  # flat
    gray_2d = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    edges = sobel(gray_2d)
    edge_density = float(edges[circle_mask].mean())

    # CMYK energy distribution
    common_black = np.minimum(np.minimum(255.0 - r, 255.0 - g), 255.0 - b)
    c_energy = float(((255.0 - r) - common_black)[circle_mask].sum())
    m_energy = float(((255.0 - g) - common_black)[circle_mask].sum())
    y_energy = float(((255.0 - b) - common_black)[circle_mask].sum())
    k_energy = float(common_black[circle_mask].sum())
    total_energy = c_energy + m_energy + y_energy + k_energy

    # Proportions
    if total_energy > 0:
        c_pct = c_energy / total_energy * 100
        m_pct = m_energy / total_energy * 100
        y_pct = y_energy / total_energy * 100
        k_pct = k_energy / total_energy * 100
    else:
        c_pct = m_pct = y_pct = k_pct = 25.0

    # Dark area ratio
    dark_ratio = float((luminance < 80).sum()) / len(luminance)
    # Light area ratio
    light_ratio = float((luminance > 200).sum()) / len(luminance)

    # ─── Generate suggestions ───
    # Mode suggestion
    color_energy_ratio = (c_energy + m_energy + y_energy) / max(total_energy, 1)
    if color_energy_ratio < 0.15:
        suggested_mode = "K Only"
        mode_reason = "Gambar didominasi grayscale/hitam, warna CMY minimal."
    else:
        suggested_mode = "CMYK"
        mode_reason = f"Gambar punya kontribusi warna {color_energy_ratio:.0%} dari total."

    # Pin count suggestion — juga tergantung ukuran frame
    # Frame besar bisa tampung lebih banyak pin
    size_factor = frame_diameter_cm / 50.0  # normalize ke 50cm
    if edge_density > 0.08:
        base_pins = 360
        pin_reason = "Banyak detail halus, butuh pin lebih banyak."
    elif edge_density > 0.04:
        base_pins = 300
        pin_reason = "Detail sedang."
    else:
        base_pins = 240
        pin_reason = "Gambar relatif smooth/simple."
    # Scale pins berdasarkan ukuran frame
    suggested_pins = int(base_pins * max(0.7, min(1.3, size_factor)))
    suggested_pins = max(120, min(480, suggested_pins))
    pin_reason += f" (scaled untuk frame {frame_diameter_cm}cm)"

    # Line count suggestion — frame besar butuh lebih banyak garis
    if dark_ratio > 0.5:
        base_lines = 6000
        line_reason = "Banyak area gelap yang perlu di-cover."
    elif dark_ratio > 0.3:
        base_lines = 5000
        line_reason = "Kegelapan sedang."
    else:
        base_lines = 4000
        line_reason = "Gambar cukup terang, tidak perlu terlalu banyak garis."
    suggested_lines = int(base_lines * max(0.7, min(1.5, size_factor)))
    line_reason += f" (scaled untuk {frame_diameter_cm}cm)"

    # Highlight protection
    if light_ratio > 0.3:
        suggested_highlight = 25
        hl_reason = "Banyak area terang yang perlu dilindungi."
    elif light_ratio > 0.15:
        suggested_highlight = 18
        hl_reason = "Area terang sedang."
    else:
        suggested_highlight = 12
        hl_reason = "Sedikit area terang."

    # Edge boost
    if edge_density > 0.06:
        suggested_edge = 1.5
        edge_reason = "Sudah banyak edge, tidak perlu boost tinggi."
    else:
        suggested_edge = 2.0
        edge_reason = "Edge sedikit, boost lebih tinggi untuk definisi."

    # Resolution
    if edge_density > 0.06:
        suggested_res = 550
    else:
        suggested_res = 450

    # Stride suggestion
    if suggested_mode == "K Only":
        suggested_stride = 3
        stride_reason = "K-only: stride besar untuk menghindari moiré pattern."
    elif edge_density > 0.06:
        suggested_stride = 1
        stride_reason = "Detail tinggi: stride kecil agar tidak miss detail."
    else:
        suggested_stride = 2
        stride_reason = "Detail sedang, stride 2 cukup."

    # Recent pins buffer
    if suggested_mode == "K Only":
        suggested_recent = 50
        recent_reason = "K-only: buffer besar mencegah pola repetitif/moiré."
    elif dark_ratio > 0.4:
        suggested_recent = 25
        recent_reason = "Area gelap banyak, buffer sedang untuk diversity."
    else:
        suggested_recent = 15
        recent_reason = "Gambar terang, buffer standar cukup."

    # Min line length (pixel)
    base_res = suggested_res
    if suggested_mode == "K Only":
        suggested_min_line = max(15, base_res // 5)
        minline_reason = "K-only: garis pendek bikin sisik, tolak yang < 20% diameter."
    elif edge_density > 0.06:
        suggested_min_line = max(10, base_res // 10)
        minline_reason = "Detail banyak: izinkan garis agak pendek untuk presisi."
    else:
        suggested_min_line = max(10, base_res // 8)
        minline_reason = "Standar: tolak garis < 12.5% diameter."

    return {
        "avg_brightness": avg_brightness,
        "brightness_std": brightness_std,
        "avg_saturation": avg_saturation,
        "edge_density": edge_density,
        "dark_ratio": dark_ratio,
        "light_ratio": light_ratio,
        "c_pct": c_pct,
        "m_pct": m_pct,
        "y_pct": y_pct,
        "k_pct": k_pct,
        "suggested_mode": suggested_mode,
        "mode_reason": mode_reason,
        "suggested_pins": suggested_pins,
        "pin_reason": pin_reason,
        "suggested_lines": suggested_lines,
        "line_reason": line_reason,
        "suggested_highlight": suggested_highlight,
        "hl_reason": hl_reason,
        "suggested_edge": suggested_edge,
        "edge_reason": edge_reason,
        "suggested_res": suggested_res,
        "suggested_stride": suggested_stride,
        "stride_reason": stride_reason,
        "suggested_recent": suggested_recent,
        "recent_reason": recent_reason,
        "suggested_min_line": suggested_min_line,
        "minline_reason": minline_reason,
        "frame_diameter_cm": frame_diameter_cm,
    }


# ═══════════════════════════ TARGET GENERATION ═══════════════════════════


def rgb_to_thread_targets_cmyk(img_arr, highlight_threshold):
    """CMYK separation dengan highlight protection."""
    img_float = img_arr.astype(np.float32)
    r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    common_black = np.minimum(np.minimum(255.0 - r, 255.0 - g), 255.0 - b)
    c_target = (255.0 - r) - common_black
    m_target = (255.0 - g) - common_black
    y_target = (255.0 - b) - common_black
    k_target = common_black

    for t in (c_target, m_target, y_target, k_target):
        t[t < highlight_threshold] = 0.0

    return {
        "C": np.clip(c_target, 0.0, 255.0),
        "M": np.clip(m_target, 0.0, 255.0),
        "Y": np.clip(y_target, 0.0, 255.0),
        "K": np.clip(k_target, 0.0, 255.0),
    }


def rgb_to_thread_targets_k_only(img_arr, highlight_threshold):
    """Grayscale (K only) — konversi ke luminance inverted."""
    img_float = img_arr.astype(np.float32)
    r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    k_target = 255.0 - luminance
    k_target[k_target < highlight_threshold] = 0.0

    return {
        "C": np.zeros_like(k_target),
        "M": np.zeros_like(k_target),
        "Y": np.zeros_like(k_target),
        "K": np.clip(k_target, 0.0, 255.0),
    }


def apply_mask_to_targets(targets, mask, color_order):
    masked_targets = {}
    for color in color_order:
        channel = targets[color].copy()
        channel[~mask] = 0.0
        masked_targets[color] = channel
    return masked_targets


# ═══════════════════════════ CORE ALGORITHM ═══════════════════════════


def build_dynamic_budget(targets, total_lines, color_order):
    """Budget dinamis berdasarkan energi tiap channel."""
    energy = {color: float(targets[color].sum()) for color in color_order}
    total_energy = sum(energy.values())

    if total_energy <= 0:
        base = total_lines // len(color_order)
        return {color: base for color in color_order}

    raw_budget = {
        color: (energy[color] / total_energy) * total_lines for color in color_order
    }
    budget = {color: int(math.floor(raw_budget[color])) for color in color_order}

    assigned = sum(budget.values())
    remaining_slots = total_lines - assigned
    remainders = sorted(
        color_order,
        key=lambda color: (raw_budget[color] - budget[color]),
        reverse=True,
    )
    for idx in range(remaining_slots):
        budget[remainders[idx % len(remainders)]] += 1

    active_colors = [color for color in color_order if energy[color] > 0]
    if active_colors and total_lines >= len(active_colors):
        for color in active_colors:
            if budget[color] == 0:
                donor = max(color_order, key=lambda item: budget[item])
                if budget[donor] > 1:
                    budget[donor] -= 1
                    budget[color] += 1

    return budget


def compute_adaptive_stride(num_pins, min_pin_gap):
    usable_range = num_pins - 2 * min_pin_gap
    if usable_range <= 0:
        return 1
    max_reasonable_stride = max(1, usable_range // 25)
    return max(1, min(3, max_reasonable_stride))


def auto_min_pin_gap(num_pins, k_only=False):
    # K-only butuh gap lebih besar untuk hindari moiré
    if k_only:
        return max(15, num_pins // 12)
    return max(8, num_pins // 24)


def precalculate_lines(pins, resolution, min_pin_gap, status_text=None):
    """Pre-calculate semua garis dengan progress feedback."""
    num_pins = len(pins)
    line_cache = [None] * (num_pins * num_pins)
    line_lengths = [0] * (num_pins * num_pins)

    total_pairs = 0
    for pin_a in range(num_pins):
        total_pairs += max(0, num_pins - pin_a - min_pin_gap)

    computed = 0
    for pin_a in range(num_pins):
        for pin_b in range(pin_a + min_pin_gap, num_pins):
            rr, cc = line_nd(pins[pin_a], pins[pin_b], endpoint=True)
            rr = np.clip(rr.astype(np.int16), 0, resolution - 1)
            cc = np.clip(cc.astype(np.int16), 0, resolution - 1)
            pixel_len = len(rr)
            line_cache[pin_a * num_pins + pin_b] = (rr, cc)
            line_cache[pin_b * num_pins + pin_a] = (rr, cc)
            line_lengths[pin_a * num_pins + pin_b] = pixel_len
            line_lengths[pin_b * num_pins + pin_a] = pixel_len
            computed += 1

        if status_text is not None and pin_a % 15 == 0:
            pct = computed / max(total_pairs, 1)
            status_text.text(
                f"⏳ Menghitung garis... {computed}/{total_pairs} ({pct:.0%})"
            )

    if status_text is not None:
        status_text.text(f"✅ {computed} garis dihitung. Mulai generating...")

    return line_cache, line_lengths


def compute_edge_map(target_channel, edge_boost):
    base_edge = sobel(target_channel / 255.0).astype(np.float32)
    if float(base_edge.max()) <= 0.0:
        return np.zeros_like(base_edge)
    positive_vals = base_edge[base_edge > 0]
    if len(positive_vals) == 0:
        return np.zeros_like(base_edge)
    p95 = np.percentile(positive_vals, 95)
    if p95 > 0:
        base_edge = base_edge / p95
    np.clip(base_edge, 0.0, 1.5, out=base_edge)
    return base_edge * edge_boost


def compute_emptiness_map(target_channel):
    max_val = float(target_channel.max())
    if max_val <= 0:
        return np.ones_like(target_channel, dtype=np.float32)
    emptiness = 1.0 - (target_channel / max_val)
    emptiness = np.power(emptiness, 2.0)
    return emptiness.astype(np.float32)


def pick_best_pin(
    current_pin, num_pins, error_map, edge_map, emptiness_map,
    saturation_map, min_pin_gap, candidate_stride, recent_pins,
    line_cache, line_lengths, line_strength,
    over_draw_penalty, empty_crossing_penalty, min_line_length=0,
):
    best_pin = -1
    best_score = -math.inf
    best_coords = None
    best_length = 0

    for offset in range(min_pin_gap, num_pins - min_pin_gap, candidate_stride):
        test_pin = (current_pin + offset) % num_pins
        if test_pin in recent_pins:
            continue

        idx = current_pin * num_pins + test_pin
        coords = line_cache[idx]
        if coords is None:
            continue
        rr, cc = coords
        pixel_len = line_lengths[idx]
        if pixel_len < max(1, min_line_length):
            continue

        error_values = error_map[rr, cc]
        edge_values = edge_map[rr, cc]
        sat_values = saturation_map[rr, cc]
        empty_values = emptiness_map[rr, cc]

        positive = np.minimum(error_values, line_strength) * (1.0 + edge_values)
        penalty_overdraw = sat_values * over_draw_penalty
        penalty_empty = empty_values * empty_crossing_penalty * line_strength

        net = positive - penalty_overdraw - penalty_empty
        raw_score = float(net.sum())

        # Normalize: sqrt(pixel_len) instead of pixel_len
        # Ini mengurangi advantage garis pendek vs panjang (anti-moiré)
        score = raw_score / math.sqrt(pixel_len)

        if score > best_score:
            best_score = score
            best_pin = test_pin
            best_coords = (rr, cc)
            best_length = pixel_len

    return best_pin, best_score, best_coords, best_length


def render_channel_preview(channel_map):
    return Image.fromarray(
        np.clip(255.0 - channel_map, 0, 255).astype(np.uint8), mode="L"
    )


def composite_cmyk_to_rgb(coverage, mask):
    c = np.clip(coverage.get("C", np.zeros_like(mask, dtype=np.float32)), 0, 255) / 255.0
    m = np.clip(coverage.get("M", np.zeros_like(mask, dtype=np.float32)), 0, 255) / 255.0
    y = np.clip(coverage.get("Y", np.zeros_like(mask, dtype=np.float32)), 0, 255) / 255.0
    k = np.clip(coverage.get("K", np.zeros_like(mask, dtype=np.float32)), 0, 255) / 255.0

    r = 255.0 * (1.0 - c) * (1.0 - k)
    g = 255.0 * (1.0 - m) * (1.0 - k)
    b = 255.0 * (1.0 - y) * (1.0 - k)

    rgb = np.stack([r, g, b], axis=-1)
    rgb[~mask] = 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def pixel_length_to_mm(pixel_len, resolution, frame_diameter_mm):
    mm_per_pixel = frame_diameter_mm / resolution
    return pixel_len * mm_per_pixel


# ═══════════════════════════ MAIN GENERATOR ═══════════════════════════


def generate_string_art(
    targets, pins, resolution, total_lines, min_pin_gap,
    thread_thickness, preview_every, edge_boost, frame_diameter_mm,
    color_order, candidate_stride, recent_buffer_size, min_line_length,
    progress_bar, status_text, image_placeholder,
):
    num_pins = len(pins)
    current_pins = {color: 0 for color in color_order}
    recent_pins = {color: deque(maxlen=recent_buffer_size) for color in color_order}
    master_sequence = []
    circle_mask = create_circle_mask(resolution)

    effective_stride = candidate_stride

    line_cache, line_lengths = precalculate_lines(
        pins, resolution, min_pin_gap, status_text
    )

    quota = build_dynamic_budget(targets, total_lines, color_order)
    remaining = {color: quota[color] for color in color_order}
    exhausted = set()
    turns_taken = {color: 0 for color in color_order}

    strength_scale = max(thread_thickness / 0.2, 0.5)
    base_strength = 20.0 * strength_scale
    line_strengths = {
        color: base_strength * CHANNEL_STRENGTH_MULTIPLIER.get(color, 1.0)
        for color in color_order
    }

    over_draw_penalty = 0.4
    empty_crossing_penalty = 0.5

    error_maps = {}
    edge_maps = {}
    emptiness_maps = {}
    coverage = {}
    saturation_maps = {}

    for color in color_order:
        coverage[color] = np.zeros((resolution, resolution), dtype=np.float32)
        saturation_maps[color] = np.zeros((resolution, resolution), dtype=np.float32)
        error_maps[color] = targets[color].astype(np.float32).copy()
        edge_maps[color] = compute_edge_map(targets[color], edge_boost)
        emptiness_maps[color] = compute_emptiness_map(targets[color])

    step = 0
    while step < total_lines:
        eligible_colors = [
            color for color in color_order
            if remaining[color] > 0 and color not in exhausted
        ]
        if not eligible_colors:
            break

        eligible_colors.sort(key=lambda c: turns_taken[c] / max(quota[c], 1))

        best_choice = None
        for color in eligible_colors:
            best_pin, best_score, best_coords, best_length = pick_best_pin(
                current_pins[color], num_pins,
                error_maps[color], edge_maps[color],
                emptiness_maps[color], saturation_maps[color],
                min_pin_gap, effective_stride, recent_pins[color],
                line_cache, line_lengths, line_strengths[color],
                over_draw_penalty, empty_crossing_penalty,
                min_line_length,
            )

            if best_pin == -1 or best_coords is None or best_score <= 0.01:
                exhausted.add(color)
                continue

            best_choice = {
                "color": color, "pin": best_pin, "score": best_score,
                "coords": best_coords, "length_px": best_length,
            }
            break

        if best_choice is None:
            break

        color = best_choice["color"]
        best_pin = best_choice["pin"]
        rr, cc = best_choice["coords"]
        start_pin = current_pins[color]
        pixel_len = best_choice["length_px"]
        line_mm = pixel_length_to_mm(pixel_len, resolution, frame_diameter_mm)

        strength = line_strengths[color]
        current_cov = coverage[color][rr, cc]
        target_vals = targets[color][rr, cc]
        fill_ratio = np.clip(current_cov / np.maximum(target_vals, 1.0), 0.0, 1.0)
        effective_strength = strength * (1.0 - 0.5 * fill_ratio)

        coverage[color][rr, cc] = np.clip(current_cov + effective_strength, 0.0, 255.0)
        error_maps[color][rr, cc] -= effective_strength
        np.clip(error_maps[color], 0.0, 255.0, out=error_maps[color])

        over = coverage[color][rr, cc] - targets[color][rr, cc]
        saturation_maps[color][rr, cc] = np.clip(over, 0.0, 255.0)

        start_coord = pins[start_pin]
        end_coord = pins[best_pin]
        master_sequence.append({
            "STEP": step + 1, "COLOR": color,
            "START_PIN": start_pin, "END_PIN": best_pin,
            "START_ROW": start_coord[0], "START_COL": start_coord[1],
            "END_ROW": end_coord[0], "END_COL": end_coord[1],
            "LENGTH_PX": pixel_len, "LENGTH_MM": round(line_mm, 2),
            "SCORE": round(best_choice["score"], 4),
        })

        current_pins[color] = best_pin
        recent_pins[color].append(start_pin)
        recent_pins[color].append(best_pin)
        remaining[color] -= 1
        turns_taken[color] += 1
        step += 1

        if step % preview_every == 0 or step == total_lines:
            composite_preview = composite_cmyk_to_rgb(coverage, circle_mask)
            progress_bar.progress(step / total_lines)
            parts = " ".join(f"{c}:{turns_taken[c]}" for c in color_order)
            status_text.text(
                f"🧵 Generating {step}/{total_lines} | {parts} | Stride:{effective_stride}"
            )
            image_placeholder.image(
                composite_preview,
                caption="Preview",
                use_container_width=True,
            )

    progress_bar.progress(1.0 if total_lines else 0.0)
    final_canvas = composite_cmyk_to_rgb(coverage, circle_mask)
    return master_sequence, final_canvas, coverage, quota, remaining, line_strengths


# ═══════════════════════════ STREAMLIT UI ═══════════════════════════

st.set_page_config(layout="wide", page_title="String Art Generator v4")
st.title("🧵 String Art Generator v4")
st.caption("CMYK / K-only mode, image analysis & auto-suggestion, weighted round-robin.")

with st.expander("📖 Cara Pakai & Tips", expanded=False):
    st.markdown("""
**Langkah Dasar:**
1. Upload gambar (PNG/JPG) & pilih **diameter frame** (20-60cm)
2. Sistem otomatis menganalisis gambar dan memberi **Recommended Settings** sesuai gambar + ukuran frame
3. Pilih mode warna: **CMYK** (4 warna benang) atau **K Only** (hitam saja)
4. Checkbox "Gunakan suggested settings" ON = pakai suggestion langsung, OFF = edit manual
5. Klik **🚀 Generate** — tunggu progress selesai
6. Download CSV sebagai instruksi robot/manual

---

**Penjelasan Parameter:**

| Parameter | Fungsi |
|-----------|--------|
| **Jumlah paku** | Semakin banyak = detail lebih halus, tapi proses lebih lama |
| **Total tarikan benang** | Jumlah garis total. Gambar gelap butuh lebih banyak |
| **Resolusi kerja** | Resolusi internal kalkulasi. 400-550 biasanya cukup |
| **Edge boost** | Prioritas area tepi/kontur. Tinggi = garis lebih fokus di edge |
| **Highlight protection** | Pixel target di bawah angka ini dianggap "kosong". Naikkan jika area terang (kulit/putih) masih kena garis |
| **Diameter frame** | Pilih ukuran frame (20-60cm). Mempengaruhi suggestion & kalkulasi benang |

---

**Advanced (Anti-Moiré):**

| Parameter | Fungsi | Gejala kalau salah |
|-----------|--------|--------------------|
| **Candidate stride** | Lompatan saat scan pin kandidat | Kecil = presisi tapi bisa moiré. Besar = diverse |
| **Recent pins buffer** | Jumlah pin terakhir yang di-blacklist | Kecil = pin dipakai ulang → pola repetitif/sisik |
| **Min line length** | Tolak garis lebih pendek dari ini | Kecil = banyak garis pendek → efek sisik/moiré |

---

**Troubleshooting:**

- **Hasil ada "sisik"/moiré spiral** → Naikkan min line length & recent buffer
- **Wajah terlalu gelap / area terang kena garis** → Naikkan highlight protection (20-30)
- **Warna tidak merata** → Pastikan mode CMYK, cek distribusi warna di analisis
- **Proses freeze di awal** → Normal, tunggu "Menghitung garis..." selesai (bisa 10-30 detik)
- **Hasil terlalu jarang/tipis** → Naikkan total tarikan benang atau turunkan highlight protection
- **Garis terlalu "kasar"** → Naikkan jumlah paku atau resolusi
""")

left_col, right_col = st.columns([1, 2])

# Preset diameter options (cm)
FRAME_SIZES_CM = [20, 30, 40, 50, 60]

with left_col:
    st.subheader("📷 Upload & Frame")
    uploaded_file = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg"])
    frame_diameter_cm = st.selectbox(
        "Diameter frame (cm)",
        options=FRAME_SIZES_CM,
        index=3,  # default 50cm
        format_func=lambda x: f"{x} cm ({x*10} mm)",
        help="Pilih ukuran fisik frame. Mempengaruhi suggestion parameter.",
    )
    frame_diameter_mm = frame_diameter_cm * 10

# ─── Image Analysis & Suggestions ───
analysis = None
if uploaded_file is not None:
    # Analisis di resolusi kecil untuk cepat
    analysis_img = prepare_square_image(uploaded_file, 256)
    analysis_arr = np.array(analysis_img)
    analysis_mask = create_circle_mask(256)
    analysis = analyze_image(analysis_arr, analysis_mask, frame_diameter_cm)

    with right_col:
        st.subheader("🔍 Analisis Gambar")
        preview_img = prepare_square_image(uploaded_file, 320)
        st.image(preview_img, caption="Preview", use_container_width=True)

        # CMYK channel preview
        st.markdown("**Dekomposisi CMYK:**")
        prev_targets = rgb_to_thread_targets_cmyk(analysis_arr, 15.0)
        ch_cols = st.columns(4)
        for idx, color in enumerate(COLOR_ORDER_CMYK):
            with ch_cols[idx]:
                ch_img = np.clip(prev_targets[color][:, :] / 255.0 * 255, 0, 255).astype(np.uint8)
                st.image(ch_img, caption=f"{color} ({analysis[f'{color.lower()}_pct']:.0f}%)",
                         use_container_width=True)

        # Suggestion box
        st.markdown("---")
        st.subheader(f"💡 Recommended Settings (frame {frame_diameter_cm}cm)")
        sug_col1, sug_col2 = st.columns(2)
        with sug_col1:
            st.metric("Mode", analysis["suggested_mode"])
            st.caption(analysis["mode_reason"])
            st.metric("Jumlah Paku", analysis["suggested_pins"])
            st.caption(analysis["pin_reason"])
            st.metric("Total Garis", analysis["suggested_lines"])
            st.caption(analysis["line_reason"])
        with sug_col2:
            st.metric("Highlight Protection", analysis["suggested_highlight"])
            st.caption(analysis["hl_reason"])
            st.metric("Edge Boost", f"{analysis['suggested_edge']:.1f}")
            st.caption(analysis["edge_reason"])
            st.metric("Resolusi", analysis["suggested_res"])
            st.metric("Stride", analysis["suggested_stride"])
            st.caption(analysis["stride_reason"])
            st.metric("Recent Buffer", analysis["suggested_recent"])
            st.caption(analysis["recent_reason"])
            st.metric("Min Line Length (px)", analysis["suggested_min_line"])
            st.caption(analysis["minline_reason"])

        # Stats
        with st.expander("📊 Detail Analisis"):
            st.markdown(
                f"- **Brightness**: avg={analysis['avg_brightness']:.0f}, "
                f"std={analysis['brightness_std']:.0f}\n"
                f"- **Saturation**: {analysis['avg_saturation']:.1f}\n"
                f"- **Edge density**: {analysis['edge_density']:.4f}\n"
                f"- **Dark area**: {analysis['dark_ratio']:.0%}\n"
                f"- **Light area**: {analysis['light_ratio']:.0%}\n"
                f"- **CMYK split**: C={analysis['c_pct']:.1f}% M={analysis['m_pct']:.1f}% "
                f"Y={analysis['y_pct']:.1f}% K={analysis['k_pct']:.1f}%"
            )


# ─── Configuration Panel ───
with left_col:
    if analysis is not None:
        st.markdown("---")
        st.subheader("⚙️ Parameter")

        color_mode = st.radio(
            "Mode Warna",
            ["CMYK", "K Only"],
            index=0 if analysis["suggested_mode"] == "CMYK" else 1,
            help="CMYK = 4 warna benang, K Only = hitam saja (grayscale)",
        )

        use_suggestions = st.checkbox("Gunakan suggested settings", value=True)

        if use_suggestions:
            num_pins = analysis["suggested_pins"]
            num_lines = analysis["suggested_lines"]
            resolution = analysis["suggested_res"]
            edge_boost = analysis["suggested_edge"]
            highlight_prot = analysis["suggested_highlight"]
            candidate_stride = analysis["suggested_stride"]
            recent_buffer_size = analysis["suggested_recent"]
            min_line_len = analysis["suggested_min_line"]
            st.text(
                f"Pins: {num_pins} | Lines: {num_lines} | "
                f"Res: {resolution} | Edge: {edge_boost} | HL: {highlight_prot}"
            )
            st.text(
                f"Stride: {candidate_stride} | Recent: {recent_buffer_size} | "
                f"MinLine: {min_line_len}px"
            )
        else:
            num_pins = st.number_input(
                "Jumlah paku", min_value=60, max_value=600,
                value=analysis["suggested_pins"], step=10
            )
            num_lines = st.number_input(
                "Total tarikan benang", min_value=100, max_value=15000,
                value=analysis["suggested_lines"], step=100
            )
            resolution = st.number_input(
                "Resolusi kerja", min_value=200, max_value=900,
                value=analysis["suggested_res"], step=20
            )
            edge_boost = st.slider(
                "Prioritas detail tepi", min_value=0.5, max_value=3.0,
                value=analysis["suggested_edge"], step=0.1
            )
            highlight_prot = st.slider(
                "Highlight protection", min_value=0, max_value=50,
                value=analysis["suggested_highlight"], step=2,
            )
            st.markdown("**Advanced (anti-moiré):**")
            candidate_stride = st.number_input(
                "Candidate stride",
                min_value=1, max_value=10,
                value=analysis["suggested_stride"], step=1,
                help="Lompatan saat scan pin kandidat. Besar = lebih diverse, kecil = lebih presisi.",
            )
            recent_buffer_size = st.number_input(
                "Recent pins buffer",
                min_value=5, max_value=100,
                value=analysis["suggested_recent"], step=5,
                help="Jumlah pin terakhir yang tidak boleh dipakai ulang. Besar = kurang moiré.",
            )
            min_line_len = st.number_input(
                "Min line length (px)",
                min_value=0, max_value=300,
                value=analysis["suggested_min_line"], step=5,
                help="Garis lebih pendek dari ini ditolak. Besar = kurang sisik/moiré.",
            )

        thread_thickness = st.number_input(
            "Ketebalan benang (mm)", min_value=0.1, max_value=2.0, value=0.2, step=0.1
        )
        st.text(f"Frame: {frame_diameter_cm}cm ({frame_diameter_mm}mm)")
        use_auto_gap = st.checkbox("Auto min pin gap", value=True)
        if use_auto_gap:
            is_k_only = (color_mode == "K Only")
            min_pin_gap = auto_min_pin_gap(int(num_pins), k_only=is_k_only)
            st.text(f"Auto gap: {min_pin_gap}")
        else:
            min_pin_gap = st.number_input(
                "Min pin gap", min_value=4, max_value=80, value=15, step=1
            )
        preview_every = st.number_input(
            "Preview interval", min_value=5, max_value=500, value=250, step=5
        )

        generate_btn = st.button("🚀 Generate")
    else:
        generate_btn = False


# ─── Generate ───
if uploaded_file is not None and generate_btn:
    color_order = COLOR_ORDER_CMYK if color_mode == "CMYK" else COLOR_ORDER_K

    with right_col:
        st.markdown("---")
        st.subheader("🔄 Proses")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        image_placeholder = st.empty()

        status_text.text("⏳ Mempersiapkan gambar...")

        prepared_img = prepare_square_image(uploaded_file, int(resolution))
        img_arr = np.array(prepared_img)
        circle_mask = create_circle_mask(int(resolution))

        # Target berdasarkan mode
        hl_thresh = float(highlight_prot)
        if color_mode == "CMYK":
            raw_targets = rgb_to_thread_targets_cmyk(img_arr, hl_thresh)
        else:
            raw_targets = rgb_to_thread_targets_k_only(img_arr, hl_thresh)

        targets = apply_mask_to_targets(raw_targets, circle_mask, color_order)
        pins = get_pin_coordinates(
            int(num_pins),
            (int(resolution) // 2) - 10,
            (int(resolution) // 2, int(resolution) // 2),
        )

        (
            master_sequence, final_canvas, coverage,
            quota, remaining, line_strengths,
        ) = generate_string_art(
            targets=targets, pins=pins,
            resolution=int(resolution), total_lines=int(num_lines),
            min_pin_gap=int(min_pin_gap), thread_thickness=float(thread_thickness),
            preview_every=int(preview_every), edge_boost=float(edge_boost),
            frame_diameter_mm=float(frame_diameter_mm),
            color_order=color_order,
            candidate_stride=int(candidate_stride),
            recent_buffer_size=int(recent_buffer_size),
            min_line_length=int(min_line_len),
            progress_bar=progress_bar, status_text=status_text,
            image_placeholder=image_placeholder,
        )

        # ─── Results ───
        df_sequence = pd.DataFrame(master_sequence)
        csv_data = df_sequence.to_csv(index=False)
        used_lines = len(master_sequence)

        st.success(f"✅ Selesai. Terpakai {used_lines} garis dari target {int(num_lines)}.")
        if used_lines < int(num_lines):
            st.warning(
                "Sebagian kuota berhenti lebih awal karena area warna sudah hampir habis."
            )

        # Visual comparison
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.image(prepared_img, caption="Gambar target", use_container_width=True)
        with result_col2:
            st.image(final_canvas, caption="Simulasi string art", use_container_width=True)

        # Summary table
        st.subheader("📊 Distribusi Warna & Kebutuhan Benang")
        summary_data = []
        for color in color_order:
            used = quota[color] - remaining[color]
            color_lines = [e for e in master_sequence if e["COLOR"] == color]
            total_length_mm = sum(e["LENGTH_MM"] for e in color_lines)
            total_length_m = total_length_mm / 1000.0
            summary_data.append({
                "WARNA": color, "BUDGET": quota[color], "TERPAKAI": used,
                "SISA": remaining[color],
                "TOTAL_MM": round(total_length_mm, 1),
                "TOTAL_M": round(total_length_m, 2),
                "STRENGTH": round(line_strengths[color], 1),
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        total_thread_m = sum(row["TOTAL_M"] for row in summary_data)
        st.info(
            f"🧵 **Estimasi total benang:** {total_thread_m:.2f} meter "
            f"(frame Ø{int(frame_diameter_mm)}mm, {used_lines} garis)"
        )

        # Channel previews
        if color_mode == "CMYK":
            channel_cols = st.columns(4)
            for idx, color in enumerate(COLOR_ORDER_CMYK):
                with channel_cols[idx]:
                    st.image(
                        render_channel_preview(coverage[color]),
                        caption=f"Coverage {color}",
                        use_container_width=True,
                    )
        else:
            st.image(
                render_channel_preview(coverage["K"]),
                caption="Coverage K",
                use_container_width=True,
            )

        # Download
        st.subheader("📥 Download")
        st.download_button(
            label="Download robot_sequence.csv",
            data=csv_data,
            file_name="robot_sequence.csv",
            mime="text/csv",
        )

        # Preview CSV
        st.subheader("Preview CSV")
        st.dataframe(df_sequence.head(20), use_container_width=True)

        # Stats
        with st.expander("📈 Statistik Detail"):
            if not df_sequence.empty:
                for color in color_order:
                    color_df = df_sequence[df_sequence["COLOR"] == color]
                    if not color_df.empty:
                        avg_len = color_df["LENGTH_MM"].mean()
                        max_len = color_df["LENGTH_MM"].max()
                        min_len = color_df["LENGTH_MM"].min()
                        avg_score = color_df["SCORE"].mean()
                        st.markdown(
                            f"- **{color}**: {len(color_df)} garis | "
                            f"{min_len:.1f}–{max_len:.1f}mm (avg {avg_len:.1f}mm) | "
                            f"Score avg: {avg_score:.4f}"
                        )
