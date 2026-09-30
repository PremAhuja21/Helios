import os
import csv
from glob import glob

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

CT_DIR = r"C:\Users\Nupur\Helios\data\output"

GT_DIR = r"C:\Users\Nupur\Helios\data\output"

PRED_DIR = r"C:\Users\Nupur\Helios\data\batch_predictions"

# All individual figures + CSV go here
OUTPUT_DIR = os.path.join(
    PRED_DIR,
    "visualizations"
)

# Maximum number of patients
MAX_PATIENTS = 100


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(path):
    image = sitk.ReadImage(path)

    array = sitk.GetArrayFromImage(
        image
    )

    return image, array


# ============================================================
# FIND LARGEST TUMOR SLICE
# ============================================================

def find_largest_tumor_slice(mask):
    areas = np.sum(
        mask > 0,
        axis=(1, 2)
    )

    if np.all(areas == 0):
        return None

    return int(
        np.argmax(areas)
    )


# ============================================================
# CT DISPLAY WINDOW
# ============================================================

def normalize_ct(ct_slice):
    # Original baseline window
    ct_slice = np.clip(
        ct_slice,
        -1000,
        400
    )

    ct_slice = (
        ct_slice + 1000.0
    ) / 1400.0

    return np.clip(
        ct_slice,
        0,
        1
    )


# ============================================================
# DICE + IOU
# ============================================================

def calculate_metrics(
    gt,
    pred
):
    gt = gt > 0
    pred = pred > 0

    intersection = np.logical_and(
        gt,
        pred
    ).sum()

    union = np.logical_or(
        gt,
        pred
    ).sum()

    gt_voxels = gt.sum()

    pred_voxels = pred.sum()

    # Dice
    denominator = (
        gt_voxels + pred_voxels
    )

    if denominator > 0:
        dice = (
            2.0 * intersection
            / denominator
        )
    else:
        dice = 1.0

    # IoU
    if union > 0:
        iou = (
            intersection
            / union
        )
    else:
        iou = 1.0

    return (
        float(dice),
        float(iou),
        int(gt_voxels),
        int(pred_voxels),
        int(intersection)
    )


# ============================================================
# CREATE ONE PATIENT FIGURE
# ============================================================

def create_patient_figure(
    patient,
    ct_path,
    gt_path,
    pred_path
):
    print(
        f"\nProcessing {patient}..."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    ct_image, ct = load_nifti(
        ct_path
    )

    gt_image, gt = load_nifti(
        gt_path
    )

    pred_image, pred = load_nifti(
        pred_path
    )

    # --------------------------------------------------------
    # Align prediction to CT grid
    # --------------------------------------------------------

    if pred.shape != ct.shape:
        print(
            f"  Resampling prediction "
            f"from {pred.shape} "
            f"to {ct.shape}"
        )

        pred_image = sitk.Resample(
            pred_image,
            ct_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8
        )

        pred = sitk.GetArrayFromImage(
            pred_image
        )

    # --------------------------------------------------------
    # Check shapes
    # --------------------------------------------------------

    if (
        ct.shape != gt.shape
        or gt.shape != pred.shape
    ):
        print(
            "  ERROR: Shape mismatch"
        )

        print(
            f"  CT:  {ct.shape}"
        )

        print(
            f"  GT:  {gt.shape}"
        )

        print(
            f"  Pred: {pred.shape}"
        )

        return None

    # --------------------------------------------------------
    # Find largest tumor slice
    # --------------------------------------------------------

    key_slice = (
        find_largest_tumor_slice(
            gt
        )
    )

    if key_slice is None:
        print(
            "  Empty ground-truth mask. Skipping."
        )

        return None

    # --------------------------------------------------------
    # Extract slice
    # --------------------------------------------------------

    ct_slice = ct[
        key_slice
    ]

    gt_slice = (
        gt[key_slice] > 0
    )

    pred_slice = (
        pred[key_slice] > 0
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    (
        dice,
        iou,
        gt_voxels,
        pred_voxels,
        intersection
    ) = calculate_metrics(
        gt,
        pred
    )

    print(
        f"  Largest tumor slice: "
        f"{key_slice}"
    )

    print(
        f"  GT voxels:   "
        f"{gt_voxels}"
    )

    print(
        f"  Pred voxels: "
        f"{pred_voxels}"
    )

    print(
        f"  Dice: "
        f"{dice:.4f}"
    )

    print(
        f"  IoU:  "
        f"{iou:.4f}"
    )

    # --------------------------------------------------------
    # Prepare CT
    # --------------------------------------------------------

    ct_display = normalize_ct(
        ct_slice
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # ========================================================
    # 1. CT
    # ========================================================

    axes[0].imshow(
        ct_display,
        cmap="gray"
    )

    axes[0].set_title(
        "CT",
        fontsize=16,
        fontweight="bold"
    )

    # ========================================================
    # 2. GROUND TRUTH
    # ========================================================

    axes[1].imshow(
        ct_display,
        cmap="gray"
    )

    axes[1].imshow(
        gt_slice,
        alpha=0.55
    )

    axes[1].set_title(
        "Ground Truth",
        fontsize=16,
        fontweight="bold"
    )

    # ========================================================
    # 3. MEDSAM2 PREDICTION
    # ========================================================

    axes[2].imshow(
        ct_display,
        cmap="gray"
    )

    axes[2].imshow(
        pred_slice,
        alpha=0.55
    )

    axes[2].set_title(
        "MedSAM2 Prediction",
        fontsize=16,
        fontweight="bold"
    )

    # ========================================================
    # 4. GT VS PREDICTION
    # ========================================================

    axes[3].imshow(
        ct_display,
        cmap="gray"
    )

    # Ground-truth contour
    if np.any(gt_slice):
        axes[3].contour(
            gt_slice,
            levels=[0.5],
            linewidths=2
        )

    # Prediction contour
    if np.any(pred_slice):
        axes[3].contour(
            pred_slice,
            levels=[0.5],
            linewidths=2
        )

    axes[3].set_title(
        "GT vs Prediction",
        fontsize=16,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Remove axes
    # --------------------------------------------------------

    for ax in axes:
        ax.axis("off")

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        (
            f"{patient}\n"
            f"Largest Tumor Slice: {key_slice}   |   "
            f"Dice: {dice:.4f}   |   "
            f"IoU: {iou:.4f}"
        ),
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.88
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{patient}_visualization.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  Saved: {output_path}"
    )

    return {
        "patient": patient,
        "key_slice": key_slice,
        "dice": dice,
        "iou": iou,
        "gt_voxels": gt_voxels,
        "pred_voxels": pred_voxels,
        "intersection": intersection
    }


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find prediction files
    # --------------------------------------------------------

    prediction_files = sorted(
        glob(
            os.path.join(
                PRED_DIR,
                "*_prediction.nii.gz"
            )
        )
    )

    print(
        "========================================"
    )

    print(
        "MedSAM2 NSCLC Visualization"
    )

    print(
        "========================================"
    )

    print(
        f"Prediction files found: "
        f"{len(prediction_files)}"
    )

    # Limit to requested number
    prediction_files = prediction_files[
        :MAX_PATIENTS
    ]

    print(
        f"Patients to process: "
        f"{len(prediction_files)}"
    )

    print(
        f"Output folder:\n"
        f"{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Process patients
    # --------------------------------------------------------

    for index, pred_path in enumerate(
        prediction_files,
        1
    ):
        filename = os.path.basename(
            pred_path
        )

        patient = filename.replace(
            "_prediction.nii.gz",
            ""
        )

        ct_path = os.path.join(
            CT_DIR,
            f"{patient}_ct.nii.gz"
        )

        gt_path = os.path.join(
            GT_DIR,
            f"{patient}_tumor_mask.nii.gz"
        )

        print(
            f"\n[{index}/{len(prediction_files)}]"
        )

        # ----------------------------------------------------
        # Check CT
        # ----------------------------------------------------

        if not os.path.exists(
            ct_path
        ):
            print(
                f"  Missing CT: "
                f"{ct_path}"
            )

            continue

        # ----------------------------------------------------
        # Check GT
        # ----------------------------------------------------

        if not os.path.exists(
            gt_path
        ):
            print(
                f"  Missing GT: "
                f"{gt_path}"
            )

            continue

        # ----------------------------------------------------
        # Create visualization
        # ----------------------------------------------------

        result = create_patient_figure(
            patient,
            ct_path,
            gt_path,
            pred_path
        )

        if result is not None:
            results.append(
                result
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        "all_patient_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "Patient",
                "Key Slice",
                "Dice",
                "IoU",
                "GT Voxels",
                "Predicted Voxels",
                "Intersection"
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result["patient"],
                    result["key_slice"],
                    f'{result["dice"]:.4f}',
                    f'{result["iou"]:.4f}',
                    result["gt_voxels"],
                    result["pred_voxels"],
                    result["intersection"]
                ]
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    if len(results) > 0:
        dice_scores = [
            r["dice"]
            for r in results
        ]

        iou_scores = [
            r["iou"]
            for r in results
        ]

        print(
            "\n\n========================================"
        )

        print(
            "BATCH SUMMARY"
        )

        print(
            "========================================"
        )

        print(
            f"Successful: "
            f"{len(results)}"
        )

        print(
            f"Mean Dice: "
            f"{np.mean(dice_scores):.4f}"
        )

        print(
            f"Median Dice: "
            f"{np.median(dice_scores):.4f}"
        )

        print(
            f"Std Dice: "
            f"{np.std(dice_scores):.4f}"
        )

        print(
            f"Mean IoU: "
            f"{np.mean(iou_scores):.4f}"
        )

        print(
            f"Median IoU: "
            f"{np.median(iou_scores):.4f}"
        )

        print(
            f"Std IoU: "
            f"{np.std(iou_scores):.4f}"
        )

        print(
            "\nResults CSV:"
        )

        print(
            csv_path
        )

    else:
        print(
            "\nNo patients were successfully processed."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()


import os
import csv
from glob import glob

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

CT_DIR = r"C:\Users\Nupur\Helios\data\output"

GT_DIR = r"C:\Users\Nupur\Helios\data\output"

PRED_DIR = r"C:\Users\Nupur\Helios\data\batch_predictions"

# All individual figures + CSV go here
OUTPUT_DIR = os.path.join(
    PRED_DIR,
    "visualizations"
)

# Maximum number of patients
MAX_PATIENTS = 100


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(path):
    image = sitk.ReadImage(path)

    array = sitk.GetArrayFromImage(
        image
    )

    return image, array


# ============================================================
# FIND LARGEST TUMOR SLICE
# ============================================================

def find_largest_tumor_slice(mask):
    areas = np.sum(
        mask > 0,
        axis=(1, 2)
    )

    if np.all(areas == 0):
        return None

    return int(
        np.argmax(areas)
    )


# ============================================================
# CT DISPLAY WINDOW
# ============================================================

def normalize_ct(ct_slice):
    # Original baseline window
    ct_slice = np.clip(
        ct_slice,
        -1000,
        400
    )

    ct_slice = (
        ct_slice + 1000.0
    ) / 1400.0

    return np.clip(
        ct_slice,
        0,
        1
    )


# ============================================================
# DICE + IOU
# ============================================================

def calculate_metrics(
    gt,
    pred
):
    gt = gt > 0
    pred = pred > 0

    intersection = np.logical_and(
        gt,
        pred
    ).sum()

    union = np.logical_or(
        gt,
        pred
    ).sum()

    gt_voxels = gt.sum()

    pred_voxels = pred.sum()

    # Dice
    denominator = (
        gt_voxels + pred_voxels
    )

    if denominator > 0:
        dice = (
            2.0 * intersection
            / denominator
        )
    else:
        dice = 1.0

    # IoU
    if union > 0:
        iou = (
            intersection
            / union
        )
    else:
        iou = 1.0

    return (
        float(dice),
        float(iou),
        int(gt_voxels),
        int(pred_voxels),
        int(intersection)
    )


# ============================================================
# CREATE ONE PATIENT FIGURE
# ============================================================

def create_patient_figure(
    patient,
    ct_path,
    gt_path,
    pred_path
):
    print(
        f"\nProcessing {patient}..."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    ct_image, ct = load_nifti(
        ct_path
    )

    gt_image, gt = load_nifti(
        gt_path
    )

    pred_image, pred = load_nifti(
        pred_path
    )

    # --------------------------------------------------------
    # Align prediction to CT grid
    # --------------------------------------------------------

    if pred.shape != ct.shape:
        print(
            f"  Resampling prediction "
            f"from {pred.shape} "
            f"to {ct.shape}"
        )

        pred_image = sitk.Resample(
            pred_image,
            ct_image,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8
        )

        pred = sitk.GetArrayFromImage(
            pred_image
        )

    # --------------------------------------------------------
    # Check shapes
    # --------------------------------------------------------

    if (
        ct.shape != gt.shape
        or gt.shape != pred.shape
    ):
        print(
            "  ERROR: Shape mismatch"
        )

        print(
            f"  CT:  {ct.shape}"
        )

        print(
            f"  GT:  {gt.shape}"
        )

        print(
            f"  Pred: {pred.shape}"
        )

        return None

    # --------------------------------------------------------
    # Find largest tumor slice
    # --------------------------------------------------------

    key_slice = (
        find_largest_tumor_slice(
            gt
        )
    )

    if key_slice is None:
        print(
            "  Empty ground-truth mask. Skipping."
        )

        return None

    # --------------------------------------------------------
    # Extract slice
    # --------------------------------------------------------

    ct_slice = ct[
        key_slice
    ]

    gt_slice = (
        gt[key_slice] > 0
    )

    pred_slice = (
        pred[key_slice] > 0
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    (
        dice,
        iou,
        gt_voxels,
        pred_voxels,
        intersection
    ) = calculate_metrics(
        gt,
        pred
    )

    print(
        f"  Largest tumor slice: "
        f"{key_slice}"
    )

    print(
        f"  GT voxels:   "
        f"{gt_voxels}"
    )

    print(
        f"  Pred voxels: "
        f"{pred_voxels}"
    )

    print(
        f"  Dice: "
        f"{dice:.4f}"
    )

    print(
        f"  IoU:  "
        f"{iou:.4f}"
    )

    # --------------------------------------------------------
    # Prepare CT
    # --------------------------------------------------------

    ct_display = normalize_ct(
        ct_slice
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # ========================================================
    # 1. CT
    # ========================================================

    axes[0].imshow(
        ct_display,
        cmap="gray"
    )

    axes[0].set_title(
        "CT",
        fontsize=16,
        fontweight="bold"
    )

    # ========================================================
    # 2. GROUND TRUTH
    # ========================================================

    axes[1].imshow(
        ct_display,
        cmap="gray"
    )

    axes[1].imshow(
        gt_slice,
        alpha=0.55
    )

    axes[1].set_title(
        "Ground Truth",
        fontsize=16,
        fontweight="bold"
    )

    # ========================================================
    # 3. MEDSAM2 PREDICTION
    # ========================================================

    axes[2].imshow(
        ct_display,
        cmap="gray"
    )

    axes[2].imshow(
        pred_slice,
        alpha=0.55
    )

    axes[2].set_title(
        "MedSAM2 Prediction",
        fontsize=16,
        fontweight="bold"
    )

    # ========================================================
    # 4. GT VS PREDICTION
    # ========================================================

    axes[3].imshow(
        ct_display,
        cmap="gray"
    )

    # Ground-truth contour
    if np.any(gt_slice):
        axes[3].contour(
            gt_slice,
            levels=[0.5],
            linewidths=2
        )

    # Prediction contour
    if np.any(pred_slice):
        axes[3].contour(
            pred_slice,
            levels=[0.5],
            linewidths=2
        )

    axes[3].set_title(
        "GT vs Prediction",
        fontsize=16,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Remove axes
    # --------------------------------------------------------

    for ax in axes:
        ax.axis("off")

    # --------------------------------------------------------
    # Main title
    # --------------------------------------------------------

    fig.suptitle(
        (
            f"{patient}\n"
            f"Largest Tumor Slice: {key_slice}   |   "
            f"Dice: {dice:.4f}   |   "
            f"IoU: {iou:.4f}"
        ),
        fontsize=18,
        fontweight="bold"
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.88
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{patient}_visualization.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  Saved: {output_path}"
    )

    return {
        "patient": patient,
        "key_slice": key_slice,
        "dice": dice,
        "iou": iou,
        "gt_voxels": gt_voxels,
        "pred_voxels": pred_voxels,
        "intersection": intersection
    }


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find prediction files
    # --------------------------------------------------------

    prediction_files = sorted(
        glob(
            os.path.join(
                PRED_DIR,
                "*_prediction.nii.gz"
            )
        )
    )

    print(
        "========================================"
    )

    print(
        "MedSAM2 NSCLC Visualization"
    )

    print(
        "========================================"
    )

    print(
        f"Prediction files found: "
        f"{len(prediction_files)}"
    )

    # Limit to requested number
    prediction_files = prediction_files[
        :MAX_PATIENTS
    ]

    print(
        f"Patients to process: "
        f"{len(prediction_files)}"
    )

    print(
        f"Output folder:\n"
        f"{OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Process patients
    # --------------------------------------------------------

    for index, pred_path in enumerate(
        prediction_files,
        1
    ):
        filename = os.path.basename(
            pred_path
        )

        patient = filename.replace(
            "_prediction.nii.gz",
            ""
        )

        ct_path = os.path.join(
            CT_DIR,
            f"{patient}_ct.nii.gz"
        )

        gt_path = os.path.join(
            GT_DIR,
            f"{patient}_tumor_mask.nii.gz"
        )

        print(
            f"\n[{index}/{len(prediction_files)}]"
        )

        # ----------------------------------------------------
        # Check CT
        # ----------------------------------------------------

        if not os.path.exists(
            ct_path
        ):
            print(
                f"  Missing CT: "
                f"{ct_path}"
            )

            continue

        # ----------------------------------------------------
        # Check GT
        # ----------------------------------------------------

        if not os.path.exists(
            gt_path
        ):
            print(
                f"  Missing GT: "
                f"{gt_path}"
            )

            continue

        # ----------------------------------------------------
        # Create visualization
        # ----------------------------------------------------

        result = create_patient_figure(
            patient,
            ct_path,
            gt_path,
            pred_path
        )

        if result is not None:
            results.append(
                result
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        "all_patient_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow(
            [
                "Patient",
                "Key Slice",
                "Dice",
                "IoU",
                "GT Voxels",
                "Predicted Voxels",
                "Intersection"
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result["patient"],
                    result["key_slice"],
                    f'{result["dice"]:.4f}',
                    f'{result["iou"]:.4f}',
                    result["gt_voxels"],
                    result["pred_voxels"],
                    result["intersection"]
                ]
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    if len(results) > 0:
        dice_scores = [
            r["dice"]
            for r in results
        ]

        iou_scores = [
            r["iou"]
            for r in results
        ]

        print(
            "\n\n========================================"
        )

        print(
            "BATCH SUMMARY"
        )

        print(
            "========================================"
        )

        print(
            f"Successful: "
            f"{len(results)}"
        )

        print(
            f"Mean Dice: "
            f"{np.mean(dice_scores):.4f}"
        )

        print(
            f"Median Dice: "
            f"{np.median(dice_scores):.4f}"
        )

        print(
            f"Std Dice: "
            f"{np.std(dice_scores):.4f}"
        )

        print(
            f"Mean IoU: "
            f"{np.mean(iou_scores):.4f}"
        )

        print(
            f"Median IoU: "
            f"{np.median(iou_scores):.4f}"
        )

        print(
            f"Std IoU: "
            f"{np.std(iou_scores):.4f}"
        )

        print(
            "\nResults CSV:"
        )

        print(
            csv_path
        )

    else:
        print(
            "\nNo patients were successfully processed."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
