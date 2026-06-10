import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_FILE = "grayscale pic.jpg"

img = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)

kernel_sizes = [3, 5, 9]

filter_results = {
    "Averaging": [],
    "Gaussian":  [],
    "Median":    [],
}

for k in kernel_sizes:
    filter_results["Averaging"].append(cv2.blur(img, (k, k)))
    filter_results["Gaussian"].append(cv2.GaussianBlur(img, (k, k), sigmaX=0))
    filter_results["Median"].append(cv2.medianBlur(img, k))

# ── Display all images ────────────────────────────────────────────────────────
# Layout: 4 rows (original + 3 filters) × 4 columns (original + 3 kernel sizes)
fig, axes = plt.subplots(4, 4, figsize=(16, 14))
fig.suptitle("Task B1 – Smoothing Filters Comparison", fontsize=16, fontweight="bold", y=1.01)

col_labels = ["Original", "3×3", "5×5", "9×9"]
row_labels  = ["Original"] + list(filter_results.keys())

# Row 0: original image repeated for reference
for col in range(4):
    axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[0, col].set_title(col_labels[col] if col == 0 else f"(same – {col_labels[col]})",
                           fontsize=10)
    axes[0, col].axis("off")

# Rows 1-3: filtered images
for row_idx, (filter_name, results) in enumerate(filter_results.items(), start=1):
    # Column 0: original for easy side-by-side comparison
    axes[row_idx, 0].imshow(img, cmap="gray", vmin=0, vmax=255)
    axes[row_idx, 0].set_title("Original", fontsize=10)
    axes[row_idx, 0].axis("off")

    for col_idx, (k, filtered) in enumerate(zip(kernel_sizes, results), start=1):
        axes[row_idx, col_idx].imshow(filtered, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, col_idx].set_title(f"{filter_name} {k}×{k}", fontsize=10)
        axes[row_idx, col_idx].axis("off")

# Row labels on the left
for ax, label in zip(axes[:, 0], row_labels):
    ax.set_ylabel(label, fontsize=12, fontweight="bold", rotation=90,
                  labelpad=10, va="center")

plt.tight_layout()
plt.savefig("task_b1_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved as 'task_b1_results.png'")

# ── Visual difference analysis (console) ─────────────────────────────────────
print("\n" + "="*60)
print("COMPARISON – Mean Absolute Difference from Original")
print("="*60)
print(f"{'Filter':<12} {'3×3':>10} {'5×5':>10} {'9×9':>10}")
print("-"*60)

for filter_name, results in filter_results.items():
    diffs = [np.mean(np.abs(img.astype(np.float32) - r.astype(np.float32)))
             for r in results]
    print(f"{filter_name:<12} {diffs[0]:>10.3f} {diffs[1]:>10.3f} {diffs[2]:>10.3f}")



explanation = """
 HOW KERNEL SIZE AFFECTS IMAGE DETAILS AND BLURRING

1. AVERAGING FILTER
   - Replaces each pixel with the average of its neighbours.
   - 3×3  : Mild blurring; fine edges are slightly softened.
   - 5×5  : Moderate blurring; thin lines and small details disappear.
   - 9×9  : Strong blurring; only large structures remain visible.

2. GAUSSIAN FILTER
   - Produces smoother, more natural-looking blur than averaging.
   - 3×3  : Gentle smoothing with good edge preservation.
   - 5×5  : Noticeable softening; texture reduced.
   - 9×9  : Heavy smoothing; image looks "out of focus."

3. MEDIAN FILTER
   - Excellent at removing salt-and-pepper (impulse) noise while
     preserving edges better than the other two filters.
   - 3×3  : Removes isolated noise with minimal blurring.
   - 5×5  : Stronger noise removal; some fine detail lost.
   - 9×9  : Very strong smoothing; edges start to erode.

"""
print(explanation)