import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.io import imread
from skimage.transform import resize
from skimage import exposure
import tkinter as tk
from tkinter import filedialog

# Create a Tkinter root window (it will not be shown)
root = tk.Tk()
root.withdraw()

# Open a file dialog to select the file
file_path = filedialog.askopenfilename(
    title="Select an image file",
    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
)


# Load an image
image = imread(file_path, as_gray=True)

# Compute the Radon transform
theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)

# Reconstruct the image from the sinogram
reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')

# Display the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

ax2.set_title("Radon transform\n(Sinogram)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r)

ax3.set_title("Reconstruction\n(FBP)")
ax3.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)

resized_image = resize(reconstruction_fbp, (512, 512), anti_aliasing=True)

# Ensure the image has non-negative values
rescaled_image = exposure.rescale_intensity(resized_image, out_range=(0, 1))

# Increase the brightness
# brightened_image = exposure.adjust_gamma(rescaled_image, gamma=0.2)  # gamma < 1 increases brightness

# Increase the contrast
contrast_image = exposure.equalize_hist(rescaled_image)

brightened_image = exposure.adjust_gamma(contrast_image, gamma=1.9)  # gamma < 1 increases brightness

# Save the contrast-enhanced image as a PNG file
plt.imsave('reconstruction_fbp_contrast.png', brightened_image, cmap=plt.cm.Greys_r)

plt.show()