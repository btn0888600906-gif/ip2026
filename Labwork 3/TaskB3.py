import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("grayscale pic.jpg", cv2.IMREAD_GRAYSCALE)

THRESH_1 = 100
THRESH_2 = 160

_, thresh_manual1 = cv2.threshold(img, THRESH_1, 255, cv2.THRESH_BINARY)
_, thresh_manual2 = cv2.threshold(img, THRESH_2, 255, cv2.THRESH_BINARY)
otsu_val, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
fig1.suptitle("Task B3 – Original and Histogram", fontsize=15, fontweight='bold')

axes1[0].imshow(img, cmap='gray')
axes1[0].set_title("Original Grayscale Image", fontsize=13)
axes1[0].axis('off')

axes1[1].hist(img.ravel(), bins=256, range=(0, 256), color='dimgray', edgecolor='none')
axes1[1].axvline(THRESH_1, color='dodgerblue', linewidth=2, linestyle='--', label=f'Manual T1 = {THRESH_1}')
axes1[1].axvline(THRESH_2, color='tomato',     linewidth=2, linestyle='--', label=f'Manual T2 = {THRESH_2}')
axes1[1].axvline(otsu_val, color='limegreen',  linewidth=2, linestyle='-',  label=f'Otsu T = {otsu_val:.0f}')
axes1[1].set_title("Pixel Intensity Histogram", fontsize=13)
axes1[1].set_xlabel("Pixel Value")
axes1[1].set_ylabel("Frequency")
axes1[1].legend()

plt.tight_layout()
plt.savefig("task_b3_histogram.png", dpi=150)
plt.close()

fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5))
fig2.suptitle("Task B3", fontsize=15, fontweight='bold')

for ax, image, title in zip(axes2,
    [img, thresh_manual1, thresh_manual2, thresh_otsu],
    ["Original", f"Manual T1 = {THRESH_1}", f"Manual T2 = {THRESH_2}", f"Otsu T = {otsu_val:.0f}"]):
    ax.imshow(image, cmap='gray')
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig("task_b3_result.png", dpi=150)
plt.close()

print(f"Otsu's threshold value: {otsu_val:.0f}")