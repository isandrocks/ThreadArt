import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import collections
import random
from PIL import Image, ImageDraw, ImageChops
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim

try:
    import torch_directml

    _HAS_DML = True
except ImportError:
    _HAS_DML = False

STATE_IMG_SIZE = 128


def _get_device():
    """Pick the best available device: DirectML > CUDA > CPU."""
    if _HAS_DML:
        return torch_directml.device()
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DQNetwork(nn.Module):
    """Deep Q-Network that reads a downsampled error image + current pin
    and outputs Q-values for every possible next pin."""

    def __init__(self, n_pins):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),  # 128 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),  # -> 4x4
        )
        self.pin_embed = nn.Embedding(n_pins, 64)
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_pins),
        )

    def forward(self, error_img, pin_idx):
        x = self.conv(error_img).view(error_img.size(0), -1)
        p = self.pin_embed(pin_idx)
        return self.fc(torch.cat([x, p], dim=1))


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state_img, pin, action, reward, next_img, next_pin, done):
        self.buffer.append((state_img, pin, action, reward, next_img, next_pin, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        imgs, pins, actions, rewards, next_imgs, next_pins, dones = zip(*batch)
        return (
            np.array(imgs, dtype=np.float32),
            np.array(pins, dtype=np.int64),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_imgs, dtype=np.float32),
            np.array(next_pins, dtype=np.int64),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def _downsample_error(error, to_size=STATE_IMG_SIZE):
    """Fast downsample of the error map to feed the network."""
    h, w = error.shape
    if h % to_size == 0 and w % to_size == 0:
        sh, sw = h // to_size, w // to_size
        clipped = np.clip(error, 0, 255)
        return clipped.reshape(to_size, sh, to_size, sw).mean(axis=(1, 3)).astype(np.float32) / 255.0
    else:
        img = Image.fromarray(np.clip(error, 0, 255).astype(np.uint8))
        img = img.resize((to_size, to_size), Image.Resampling.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0


def _get_action_mask(pin, n_pins, min_distance, last_pins):
    """Build a boolean mask of valid next-pin choices (True = valid)."""
    mask = np.zeros(n_pins, dtype=bool)
    for offset in range(min_distance, n_pins - min_distance):
        mask[(pin + offset) % n_pins] = True
    for p in last_pins:
        mask[p] = False
    return mask


def _setup_pins_and_lines(N_PINS, MIN_DISTANCE, length):
    """Compute pin coordinates and precompute the line pixel cache."""
    pin_coords = []
    center = length / 2
    radius = length / 2 - 0.5

    for i in range(N_PINS):
        angle = 2 * math.pi * i / N_PINS
        pin_coords.append((
            math.floor(center + radius * math.cos(angle)),
            math.floor(center + radius * math.sin(angle)),
        ))

    print("Precalculating all lines... ", end="", flush=True)
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
            line_cache_y[b * N_PINS + a] = ys
            line_cache_y[a * N_PINS + b] = ys
            line_cache_x[b * N_PINS + a] = xs
            line_cache_x[a * N_PINS + b] = xs
            line_cache_length[b * N_PINS + a] = d
            line_cache_length[a * N_PINS + b] = d
    print("done")

    return pin_coords, line_cache_x, line_cache_y, line_cache_length

def string_art_dqn_multiscale(
    N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img,
    edge_map=None, EDGE_BOOST=2.0, SSIM_TARGET=0.65,
    training_episodes=20, lr=5e-4, batch_size=128, gamma=0.9,
    train_steps_per_episode=3000, target_update_freq=300,
    replay_capacity=80000, device=None, LINE_COST=0.05,
    no_stagnation=False
):
    # Pass 1: Coarse (we reuse string_art_dqn but with fewer pins)
    coarse_pins = N_PINS // 3
    img_coarse = Image.fromarray(img).resize((128, 128), Image.Resampling.LANCZOS)
    img_coarse_np = np.array(img_coarse)
    
    print("--- DQN MULTISCALE: COARSE PASS ---")
    seq_c, res_c, ln_c, diff_c, frames_c = string_art_dqn(
        coarse_pins, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, img_coarse_np,
        edge_map=None, EDGE_BOOST=EDGE_BOOST, SSIM_TARGET=0.45,
        training_episodes=max(1, training_episodes // 2), lr=lr, batch_size=batch_size, gamma=gamma,
        train_steps_per_episode=train_steps_per_episode, target_update_freq=target_update_freq,
        replay_capacity=replay_capacity, device=device, LINE_COST=LINE_COST,
        no_stagnation=no_stagnation
    )
    
    # Scale up coarse result to original size
    length = img.shape[0]
    res_c_scaled = res_c.resize((length, length), Image.Resampling.LANCZOS)
    res_c_np = np.array(res_c_scaled, dtype=np.float64)
    
    # Compute residual: what's left to draw
    residual_error = np.clip(res_c_np - img.astype(np.float64), 0, 255)
    target_img = np.clip(255 - residual_error, 0, 255).astype(np.uint8)
    
    print("--- DQN MULTISCALE: FINE PASS ---")
    seq_f, res_f, ln_f, diff_f, frames_f = string_art_dqn(
        N_PINS, MAX_LINES, MIN_LOOP, MIN_DISTANCE, LINE_WEIGHT, SCALE, target_img,
        edge_map=edge_map, EDGE_BOOST=EDGE_BOOST, SSIM_TARGET=SSIM_TARGET,
        training_episodes=training_episodes, lr=lr, batch_size=batch_size, gamma=gamma,
        train_steps_per_episode=train_steps_per_episode, target_update_freq=target_update_freq,
        replay_capacity=replay_capacity, device=device, LINE_COST=LINE_COST,
        no_stagnation=no_stagnation
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

def string_art_dqn(
    N_PINS,
    MAX_LINES,
    MIN_LOOP,
    MIN_DISTANCE,
    LINE_WEIGHT,
    SCALE,
    img,
    edge_map=None,
    EDGE_BOOST=2.0,
    SSIM_TARGET=0.65,
    training_episodes=20,
    lr=5e-4,
    batch_size=128,
    gamma=0.9,
    train_steps_per_episode=3000,
    target_update_freq=300,
    replay_capacity=80000,
    device=None,
    LINE_COST=0.05,
    no_stagnation=False
):
    """Generate string art using a DQN-trained policy for pin selection.

    The function has two phases:
      1. **Training** – run several episodes of line placement with epsilon-greedy
         exploration, training a DQN on the (error_image, pin) -> Q-value mapping.
      2. **Inference** – use a hybrid greedy+DQN policy to produce the final artwork.
         For each step, the top-K greedy candidates are re-ranked by the DQN to
         combine short-term optimality with learned long-term value.

    Returns the same tuple as ``string_art``:
        (pin_sequence, result_image, line_number, current_absdiff, frames)
    """
    assert img.shape[0] == img.shape[1]
    length = img.shape[0]

    # Circular mask
    X, Y = np.ogrid[0:length, 0:length]
    circlemask = (X - length / 2) ** 2 + (Y - length / 2) ** 2 > (length / 2) ** 2
    img[circlemask] = 0xFF

    pin_coords, line_cache_x, line_cache_y, line_cache_length = _setup_pins_and_lines(
        N_PINS, MIN_DISTANCE, length
    )

    original_error = np.ones(img.shape, dtype=np.float64) * 0xFF - img.astype(np.float64)

    if edge_map is None:
        edge_map = sobel(img.astype(np.float64))
        if edge_map.max() > 0:
            edge_map /= edge_map.max()

    # ---- Device ----
    if device is None:
        device = _get_device()
    print(f"Using device: {device}")

    # ---- Networks ----
    policy_net = DQNetwork(N_PINS).to(device)
    target_net = DQNetwork(N_PINS).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr, foreach=False) # type: ignore
    replay = ReplayBuffer(replay_capacity)

    total_steps = 0
    epsilon_start = 1.0
    epsilon_end = 0.05

    # ==================================================================
    # TRAINING PHASE
    # ==================================================================
    print(f"\n=== DQN Training: {training_episodes} episodes ===")

    for episode in range(training_episodes):
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * episode / max(1, training_episodes - 1),
        )

        error = original_error.copy()
        pin = random.randint(0, N_PINS - 1)
        last_pins = collections.deque(maxlen=MIN_LOOP)
        episode_reward = 0.0
        steps = min(train_steps_per_episode, MAX_LINES)
        episode_loss = 0.0
        loss_count = 0

        for step in range(steps):
            state_img = _downsample_error(error)
            mask = _get_action_mask(pin, N_PINS, MIN_DISTANCE, last_pins)
            valid = np.where(mask)[0]
            if len(valid) == 0:
                break

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = int(np.random.choice(valid))
            else:
                with torch.no_grad():
                    img_t = torch.from_numpy(state_img).unsqueeze(0).unsqueeze(0).to(device)
                    pin_t = torch.LongTensor([pin]).to(device)
                    q = policy_net(img_t, pin_t).cpu().numpy()[0]
                    q[~mask] = -np.inf
                    action = int(np.argmax(q))

            xs = line_cache_x[action * N_PINS + pin]
            ys = line_cache_y[action * N_PINS + pin]
            if xs is None or ys is None:
                break

            # Reward = total error reduction from this line (clipped)
            line_error = error[ys, xs]
            reward = float(np.sum(np.minimum(line_error, LINE_WEIGHT) * (1.0 + EDGE_BOOST * edge_map[ys, xs]))) / (length * 10.0) - LINE_COST

            # Update error
            line_mask = np.zeros(error.shape, np.float64)
            line_mask[ys, xs] = LINE_WEIGHT
            error -= line_mask
            np.clip(error, 0, 255, out=error)

            next_img = _downsample_error(error)
            done = float(step == steps - 1)

            replay.push(state_img, pin, action, reward, next_img, action, done)
            episode_reward += reward
            last_pins.append(action)
            pin = action
            total_steps += 1

            # ---- train on a mini-batch every 4 environment steps ----
            if len(replay) >= batch_size and total_steps % 4 == 0:
                s_imgs, s_pins, acts, rews, ns_imgs, ns_pins, dones_b = replay.sample(batch_size)

                s_imgs_t = torch.from_numpy(s_imgs).unsqueeze(1).to(device)
                s_pins_t = torch.from_numpy(s_pins).to(device)
                acts_t = torch.from_numpy(acts).to(device)
                rews_t = torch.from_numpy(rews).to(device)
                ns_imgs_t = torch.from_numpy(ns_imgs).unsqueeze(1).to(device)
                ns_pins_t = torch.from_numpy(ns_pins).to(device)
                dones_t = torch.from_numpy(dones_b).to(device)

                q_cur = policy_net(s_imgs_t, s_pins_t).gather(1, acts_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # Double DQN: policy selects action, target evaluates it
                    best_actions = policy_net(ns_imgs_t, ns_pins_t).argmax(1, keepdim=True)
                    q_next = target_net(ns_imgs_t, ns_pins_t).gather(1, best_actions).squeeze(1)
                    q_tgt = rews_t + gamma * q_next * (1.0 - dones_t)

                loss = F.smooth_l1_loss(q_cur, q_tgt)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                episode_loss += loss.item()
                loss_count += 1

            # ---- sync target network periodically ----
            if total_steps % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        avg_loss = episode_loss / max(1, loss_count)
        print(
            f"  Episode {episode + 1}/{training_episodes} | "
            f"eps={epsilon:.3f} | reward={episode_reward:.1f} | "
            f"loss={avg_loss:.4f} | steps={step + 1}"
        )

    # ==================================================================
    # INFERENCE PHASE  (hybrid greedy + DQN)
    # ==================================================================
    print("\n=== Generating final result (hybrid greedy+DQN) ===")

    error = original_error.copy()
    result = Image.new("L", (length * SCALE, length * SCALE), 0xFF)
    draw = ImageDraw.Draw(result)

    pin = 0
    last_pins = collections.deque(maxlen=MIN_LOOP)
    pin_sequence = []
    frames = []
    line_number = 0
    current_absdiff = 0.0
    error_history = []  # rolling window for stagnation detection
    TOP_K = 20  # number of greedy candidates the DQN re-ranks

    policy_net.eval()

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

        # ---- hybrid pin selection: greedy top-K, then DQN re-rank ----
        mask = _get_action_mask(pin, N_PINS, MIN_DISTANCE, last_pins)
        valid = np.where(mask)[0]
        if len(valid) == 0:
            break

        # Compute greedy scores for all valid pins
        greedy_scores = np.full(N_PINS, -np.inf)
        for vp in valid:
            xs = line_cache_x[vp * N_PINS + pin]
            ys = line_cache_y[vp * N_PINS + pin]
            if xs is not None and ys is not None:
                greedy_scores[vp] = float(np.sum(np.minimum(error[ys, xs], LINE_WEIGHT) * (1.0 + EDGE_BOOST * edge_map[ys, xs])))

        # Pick top-K candidates by greedy score
        k = min(TOP_K, len(valid))
        top_k_pins = np.argpartition(greedy_scores, -k)[-k:]
        top_k_pins = top_k_pins[np.argsort(greedy_scores[top_k_pins])[::-1]]

        # DQN re-ranks the top-K
        state_img = _downsample_error(error)
        with torch.no_grad():
            img_t = torch.from_numpy(state_img).unsqueeze(0).unsqueeze(0).to(device)
            pin_t = torch.LongTensor([pin]).to(device)
            q = policy_net(img_t, pin_t).cpu().numpy()[0]

        # Combine: normalised greedy score + DQN Q-value
        g_max = max(greedy_scores[top_k_pins[0]], 1e-8)
        q_top = q[top_k_pins]
        q_range = max(q_top.max() - q_top.min(), 1e-8)

        combined = np.zeros(k)
        for i, tp in enumerate(top_k_pins):
            combined[i] = 0.6 * (greedy_scores[tp] / g_max) + 0.4 * ((q[tp] - q_top.min()) / q_range) # weighted sum of normalised greedy and DQN scores

        best_pin = int(top_k_pins[np.argmax(combined)])

        xs = line_cache_x[best_pin * N_PINS + pin]
        ys = line_cache_y[best_pin * N_PINS + pin]
        if xs is None or ys is None:
            break

        # ---- place the line ----
        line_mask_arr = np.zeros(error.shape, np.float64)
        line_mask_arr[ys, xs] = LINE_WEIGHT / SCALE
        error -= line_mask_arr
        np.clip(error, 0, 255, out=error)

        draw.line(
            [
                (pin_coords[pin][0] * SCALE, pin_coords[pin][1] * SCALE),
                (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE),
            ],
            fill=0,
            width=1,
        )

        frames.append(
            [
                (pin_coords[pin][0] * SCALE, pin_coords[pin][1] * SCALE),
                
                (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE),
            ]
        )

        last_pins.append(best_pin)
        pin_sequence.append(best_pin)

        pin = best_pin

    # Final absdiff computation
    img_result = result.resize((length, length), Image.Resampling.LANCZOS)
    img_result = np.array(img_result, dtype=np.float64)
    diff = np.abs(img_result - img.astype(np.float64))
    current_absdiff = diff.sum() / (length * length)

    return pin_sequence, result, line_number, current_absdiff, frames
