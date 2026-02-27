import numpy as np
from PIL import Image

width = 800
pitch = 832
height = 600

# Read raw data
data = np.fromfile("frame.raw", dtype=np.uint8)
# Reshape to (height, pitch, 4)
data = data.reshape((height, pitch, 4))
# Crop to (height, width, 4)
data = data[:, :width, :]

# Convert BGRA to RGBA (swap index 0 and 2)
rgba = np.zeros_like(data)
rgba[:, :, 0] = data[:, :, 2]
rgba[:, :, 1] = data[:, :, 1]
rgba[:, :, 2] = data[:, :, 0]
rgba[:, :, 3] = data[:, :, 3]

img = Image.fromarray(rgba, 'RGBA')
img.save("screenshot.png")
print("Saved screenshot.png")
