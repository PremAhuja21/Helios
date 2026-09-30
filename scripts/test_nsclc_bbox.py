from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# FILE PATHS
# ============================================================

CT_PATH = Path(
    r"C:\Users\Nupur\Helios\data\output\LUNG1-001_ct.nii.gz"
)

MASK_PATH = Path(
    r"C:\Users\Nupur\Helios\data\output\LUNG1-001_tumor_mask.nii.gz"
)


print("========================================")
print("NSCLC CT + BOUNDING BOX TEST")
print("========================================")


# ============================================================
# LOAD CT
# ============================================================

print("Loading CT...")

ct_image = sitk.ReadImage(str(CT_PATH))
ct = sitk.GetArrayFromImage(ct_image)

print("CT shape:", ct.shape)


# ============================================================
# LOAD TUMOR MASK
# ============================================================

print("Loading tumor mask...")

mask_image = sitk.ReadImage(str(MASK_PATH))
mask = sitk.GetArrayFromImage(mask_image)

print("Mask shape:", mask.shape)


# ============================================================
# FIND LARGEST TUMOR SLICE
# ============================================================

print("\nFinding largest tumor slice...")

tumor_pixels_per_slice = np.sum(mask > 0, axis=(1, 2))

largest_slice = int(np.argmax(tumor_pixels_per_slice))
largest_area = int(tumor_pixels_per_slice[largest_slice])

print("Largest tumor slice:", largest_slice)
print("Tumor pixels:", largest_area)


# ============================================================
# FIND BOUNDING BOX
# ============================================================

slice_mask = mask[largest_slice] > 0

ys, xs = np.where(slice_mask)

x_min = int(xs.min())
x_max = int(xs.max())
y_min = int(ys.min())
y_max = int(ys.max())

print("\nBounding box:")
print("x_min:", x_min)
print("y_min:", y_min)
print("x_max:", x_max)
print("y_max:", y_max)


# ============================================================
# GET ACTUAL CT SLICE
# ============================================================

ct_slice = ct[largest_slice]


# ============================================================
# DISPLAY CT + BOUNDING BOX
# ============================================================

print("\nOpening CT visualization...")

plt.figure(figsize=(8, 8))

plt.imshow(ct_slice, cmap="gray")

plt.gca().add_patch(
    plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        fill=False,
        edgecolor="red",
        linewidth=2,
    )
)

plt.title(
    f"LUNG1-001\n"
    f"Largest Tumor Slice: {largest_slice}\n"
    f"BBox: [{x_min}, {y_min}, {x_max}, {y_max}]"
)

plt.axis("off")
plt.show()

print("\nDONE!")