import sys
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

img = cv2.imread(sys.argv[-1], cv2.IMREAD_GRAYSCALE)

kernel_sizes = [3, 5, 9]

fig, axes = plt.subplots(4, 4, figsize=(16, 12))
fig.suptitle("Task B1 – Smoothing Filters", fontsize=14, fontweight="bold")

images = [
    ("Original", [img, img, img]),
    ("Averaging", [cv2.blur(img, (k, k)) for k in kernel_sizes]),
    ("Gaussian",  [cv2.GaussianBlur(img, (k, k), 0) for k in kernel_sizes]),
    ("Median",    [cv2.medianBlur(img, k) for k in kernel_sizes]),
]

col_titles = ["Original", "3×3", "5×5", "9×9"]

for row, (name, imgs) in enumerate(images):
    for col in range(4):
        axes[row, col].imshow(imgs[0] if col == 0 else imgs[col - 1], cmap="gray")
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(col_titles[col])
    axes[row, 0].set_ylabel(name, fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("task_b1_results.png", dpi=150, bbox_inches="tight")
print("Done")