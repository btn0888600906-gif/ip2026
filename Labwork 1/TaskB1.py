import sys
from PIL import Image
import numpy as np

img_path = sys.argv[-1]

input_img = Image.open(img_path)
pixels = input_img.load()
w, h = input_img.size
grey_img = Image.new("L", (w, h))
grey = grey_img.load()

for y in range(h):
    for x in range(w):
        r, g, b = pixels[x, y]
        luminance = int(0.2126 * r + 0.7152 * g + 0.0722 * b)
        grey[x, y] = luminance

grey_img.save("grey.jpg")

filtered_img = Image.new("L", (w, h))
filtered = filtered_img.load()

kernel = np.array([[ 1, 1, 1],
                   [ 1, 1, 1],
                   [ 1, 1, 1]], dtype = float)
kernel /= kernel.sum()

for y in range(h):
    for x in range(w):
        matrix = np.zeros((3, 3))
        value = 0
        for j in range(-1, 2):
            for i in range(-1, 2):
                if (0 <= y + j < h) and (0 <= x + i < w) :
                    matrix[i, j] = grey[x+i, y+j]
                else:
                    matrix[i, j] = 1
                value += matrix[i, j] * kernel[i, j]
        filtered[x, y] = abs(int(value))
filtered_img.save("filtered.jpg")