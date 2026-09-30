import numpy as np
import SimpleITK as sitk


GT_PATH = r"C:\Users\Nupur\Helios\data\test_nsclc\masks\LUNG1-001_tumor_mask.nii.gz"

PRED_PATH = r"C:\Users\Nupur\Helios\data\test_nsclc\predictions\LUNG1-001_prediction.nii.gz"


print("Loading ground truth...")
gt_image = sitk.ReadImage(GT_PATH)
gt = sitk.GetArrayFromImage(gt_image) > 0

print("Loading MedSAM2 prediction...")
pred_image = sitk.ReadImage(PRED_PATH)
pred = sitk.GetArrayFromImage(pred_image) > 0


print("\nShapes:")
print("Ground truth:", gt.shape)
print("Prediction:  ", pred.shape)


# Check dimensions
if gt.shape != pred.shape:
    raise ValueError(
        f"Shape mismatch: GT={gt.shape}, Prediction={pred.shape}"
    )


# Calculate voxel counts
gt_voxels = int(gt.sum())
pred_voxels = int(pred.sum())


# Intersection
intersection = int(
    np.logical_and(gt, pred).sum()
)


# Union
union = int(
    np.logical_or(gt, pred).sum()
)


# Dice coefficient
if gt_voxels + pred_voxels == 0:
    dice = 1.0
else:
    dice = (
        2.0 * intersection
        / (gt_voxels + pred_voxels)
    )


# IoU / Jaccard
if union == 0:
    iou = 1.0
else:
    iou = intersection / union


print("\n==============================")
print("MedSAM2 Evaluation")
print("==============================")

print(f"Ground-truth voxels : {gt_voxels}")
print(f"Predicted voxels    : {pred_voxels}")
print(f"Intersection        : {intersection}")
print(f"Union               : {union}")

print(f"\nDice Score          : {dice:.4f}")
print(f"IoU                 : {iou:.4f}")