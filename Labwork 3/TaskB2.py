import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("grayscale pic.jpg", cv2.IMREAD_GRAYSCALE)

noise = np.random.normal(0, 25, img.shape).astype(np.float32)
noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

smoothed = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1.5)

def edge_detect(src):
    lap = np.uint8(np.absolute(cv2.Laplacian(src, cv2.CV_64F)))
    sx  = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
    sy  = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
    sob = np.clip(np.sqrt(sx**2 + sy**2), 0, 255).astype(np.uint8)
    can = cv2.Canny(src, 50, 150)
    return lap, sob, can

lap_noisy,  sob_noisy,  can_noisy  = edge_detect(noisy)
lap_smooth, sob_smooth, can_smooth = edge_detect(smoothed)

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle("Task B2", fontsize=16, fontweight='bold')

rows = [
    ([noisy,      smoothed,     img],
     ["Noisy Image", "Smoothed Image", "Original (reference)"]),
    ([lap_noisy,  sob_noisy,  can_noisy],
     ["Laplacian (noisy)", "Sobel (noisy)", "Canny (noisy)"]),
    ([lap_smooth, sob_smooth, can_smooth],
     ["Laplacian (smoothed)", "Sobel (smoothed)", "Canny (smoothed)"]),
]

for r, (images, titles) in enumerate(rows):
    for c, (im, title) in enumerate(zip(images, titles)):
        axes[r][c].imshow(im, cmap='gray')
        axes[r][c].set_title(title, fontsize=12)
        axes[r][c].axis('off')

plt.tight_layout()
plt.savefig("task_b2_result.png", dpi=150)
plt.close()