import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("grayscale pic.jpg", cv2.IMREAD_GRAYSCALE)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian_abs = np.uint8(np.absolute(laplacian))

sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.uint8(np.sqrt(sobel_x**2 + sobel_y**2))
sobel = np.clip(sobel, 0, 255).astype(np.uint8)

canny = cv2.Canny(img, threshold1=50, threshold2=150)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Task B1", fontsize=16, fontweight='bold')

images   = [img,       laplacian_abs, sobel,   canny]
titles   = ["Original", "Laplacian",  "Sobel", "Canny"]
cmaps    = ["gray",     "gray",        "gray",  "gray"]

for ax, image, title, cmap in zip(axes, images, titles, cmaps):
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=13)
    ax.axis("off")

plt.tight_layout()
plt.savefig("task_b1_result.png", dpi=150)
plt.close()