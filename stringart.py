import collections
import math
import os
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import time
import svgwrite

MAX_LINES = 8000
N_PINS = 36 * 8
MIN_LOOP = 1
MIN_DISTANCE = 35
LINE_WEIGHT = 16
FILENAME = "00010-1925372175.png"
SCALE = 25
HOOP_DIAMETER = 0.625     # To calculate total thread length

tic = time.perf_counter()

img = Image.open(FILENAME).convert('L')
img = ImageOps.grayscale(img)
img = np.array(img)

# Didn't bother to make it work for non-square images
assert img.shape[0] == img.shape[1]
length = img.shape[0]

def disp(image):
  image.show()

# Cut away everything around a central circle
X, Y = np.ogrid[0:length, 0:length]
circlemask = (X - length/2) ** 2 + (Y - length/2) ** 2 > length/2 * length/2
img[circlemask] = 0xFF

pin_coords = []
center = length / 2
radius = length / 2 - 1/2

# Precalculate the coordinates of every pin
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

error = np.ones(img.shape) * 0xFF - img.copy()

img_result = np.ones(img.shape) * 0xFF
lse_buffer = np.ones(img.shape) * 0xFF    # Used in the unused LSE algorithm

result = Image.new('L', (img.shape[0] * SCALE, img.shape[1] * SCALE), 0xFF)
draw = ImageDraw.Draw(result)
line_mask = np.zeros(img.shape, np.float64)

line_sequence = []
pin = 0
line_sequence.append(pin)

thread_length = 0

last_pins = collections.deque(maxlen = MIN_LOOP)

# Initialize SVG drawing
svg_filename = os.path.splitext(FILENAME)[0] + "-out.svg"
dwg = svgwrite.Drawing(svg_filename, size=(length, length))
dwg.add(dwg.rect(insert=(0, 0), size=(length, length), fill="white"))

path = dwg.path(d="M {} {}".format(*pin_coords[0]), stroke="black", fill="none", stroke_width="0.15px")

for l in range(MAX_LINES):

  if l % 100 == 0:
    print("%d " % l, end='', flush=True)

    img_result = result.resize(img.shape, Image.Resampling.LANCZOS)
    img_result = np.array(img_result)

    diff = img_result - img
    mul = np.uint8(img_result < img) * 254 + 1
    absdiff = diff * mul
    print(absdiff.sum() / (length * length))

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

  line_sequence.append(best_pin)

  xs = line_cache_x[best_pin * N_PINS + pin]
  ys = line_cache_y[best_pin * N_PINS + pin]
  weight = LINE_WEIGHT * line_cache_weight[best_pin*N_PINS + pin]
  
      # Add the line to the SVG path
  path.push("L {} {}".format(*pin_coords[best_pin]))

  line_mask.fill(0)
  line_mask[ys, xs] = weight
  error = error - line_mask
  error.clip(0, 255)

  draw.line(
    [(pin_coords[pin][0] * SCALE, pin_coords[pin][1] * SCALE),
     (pin_coords[best_pin][0] * SCALE, pin_coords[best_pin][1] * SCALE)],
    fill=0, width=4)

  x0 = pin_coords[pin][0]
  y0 = pin_coords[pin][1]

  x1 = pin_coords[best_pin][0]
  y1 = pin_coords[best_pin][1]

  dist = math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0)*(y1 - y0))
  thread_length += HOOP_DIAMETER / length * dist

  last_pins.append(best_pin)
  pin = best_pin
  
dwg.add(path)
dwg.save()

img_result = result.resize(img.shape, Image.Resampling.LANCZOS)
img_result = np.array(img_result)

diff = img_result - img
mul = np.uint8(img_result < img) * 254 + 1
absdiff = diff * mul

print(absdiff.sum() / (length * length))

print('\x07')
toc = time.perf_counter()
print("%.1f seconds" % (toc - tic))

result.save(os.path.splitext(FILENAME)[0] + "-out.png")

with open(os.path.splitext(FILENAME)[0] + ".json", "w") as f:
  f.write(str(line_sequence))
