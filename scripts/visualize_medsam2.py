import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


CT_PATH = r"C:\Users\Nupur\Helios\data\test_nsclc\ct\LUNG1-001_ct.nii.gz"

GT_PATH = r"C:\Users\Nupur\Helios\data\test_nsclc\masks\LUNG1-001_tumor_mask.nii.gz"

PRED_PATH = r"C:\Users\Nupur\Helios\data\test_nsclc\predictions\LUNG1-001_prediction.nii.gz"


# Load volumes
ct = sitk.GetArrayFromImage(sitk.ReadImage(CT_PATH))
gt = sitk.GetArrayFromImage(sitk.ReadImage(GT_PATH)) > 0
pred = sitk.GetArrayFromImage(sitk.ReadImage(PRED_PATH)) > 0


# Find largest GT tumor slice
areas = gt.sum(axis=(1, 2))
slice_idx = int(np.argmax(areas))

print("Largest tumor slice:", slice_idx)
print("GT tumor pixels:", int(areas[slice_idx]))
print("Prediction pixels on slice:", int(pred[slice_idx].sum()))


# Get slices
ct_slice = ct[slice_idx]
gt_slice = gt[slice_idx]
pred_slice = pred[slice_idx]


# Normalize CT for display
ct_display = np.clip(ct_slice, -1000, 400)
ct_display = (ct_display + 1000) / 1400


# Create figure
fig, axes = plt.subplots(1, 4, figsize=(20, 5))


# 1. CT
axes[0].imshow(ct_display, cmap="gray")
axes[0].set_title("CT")
axes[0].axis("off")


# 2. Ground truth
axes[1].imshow(ct_display, cmap="gray")
axes[1].imshow(
    gt_slice,
    alpha=0.45
)
axes[1].set_title("Ground Truth")
axes[1].axis("off")


# 3. Prediction
axes[2].imshow(ct_display, cmap="gray")
axes[2].imshow(
    pred_slice,
    alpha=0.45
)
axes[2].set_title("MedSAM2 Prediction")
axes[2].axis("off")


# 4. Overlay
axes[3].imshow(ct_display, cmap="gray")

axes[3].contour(
    gt_slice,
    levels=[0.5],
    linewidths=2
)

axes[3].contour(
    pred_slice,
    levels=[0.5],
    linewidths=2
)

axes[3].set_title("GT vs Prediction")
axes[3].axis("off")


plt.tight_layout()
plt.show()