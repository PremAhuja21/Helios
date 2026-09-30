import os
import csv
from glob import glob

import numpy as np
import SimpleITK as sitk
from skimage import measure


CT_DIR = r"C:\Users\Nupur\Helios\data\output"
MASK_DIR = r"C:\Users\Nupur\Helios\data\output"

OUTPUT_DIR = r"C:\Users\Nupur\Helios\data\classical_baseline"

MAX_PATIENTS = 5

# Simple CT intensity threshold
HU_THRESHOLD = -300


def load(path):

    image = sitk.ReadImage(path)

    array = sitk.GetArrayFromImage(
        image
    )

    return image, array


def largest_cc(mask):

    labels = measure.label(
        mask
    )

    if labels.max() == 0:

        return mask.astype(
            np.uint8
        )

    largest = (
        np.argmax(
            np.bincount(
                labels.flat
            )[1:]
        ) + 1
    )

    return (
        labels == largest
    ).astype(
        np.uint8
    )


def metrics(gt, pred):

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

    dice = (
        2 * intersection
        / (gt.sum() + pred.sum())
        if gt.sum() + pred.sum() > 0
        else 1.0
    )

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    return (
        float(dice),
        float(iou)
    )


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    ct_files = sorted(
        glob(
            os.path.join(
                CT_DIR,
                "*_ct.nii.gz"
            )
        )
    )[:MAX_PATIENTS]

    results = []

    print(
        f"Running classical baseline "
        f"on {len(ct_files)} patients."
    )

    print(
        f"HU threshold: "
        f"{HU_THRESHOLD}"
    )

    for i, ct_path in enumerate(
        ct_files,
        1
    ):

        patient = (
            os.path.basename(
                ct_path
            )
            .replace(
                "_ct.nii.gz",
                ""
            )
        )

        print(
            f"\n[{i}/{len(ct_files)}] "
            f"{patient}"
        )

        mask_path = os.path.join(
            MASK_DIR,
            f"{patient}_tumor_mask.nii.gz"
        )

        _, ct = load(
            ct_path
        )

        _, gt = load(
            mask_path
        )

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        prediction = (
            ct > HU_THRESHOLD
        )

        # ----------------------------------------------------
        # Keep largest 3D component
        # ----------------------------------------------------

        prediction = largest_cc(
            prediction
        )

        dice, iou = metrics(
            gt,
            prediction
        )

        print(
            f"  Dice: {dice:.4f}"
        )

        print(
            f"  IoU:  {iou:.4f}"
        )

        results.append(
            [
                patient,
                dice,
                iou
            ]
        )

    csv_path = os.path.join(
        OUTPUT_DIR,
        "classical_baseline_results.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "patient",
                "dice",
                "iou"
            ]
        )

        writer.writerows(
            results
        )

    dice_scores = [
        r[1]
        for r in results
    ]

    iou_scores = [
        r[2]
        for r in results
    ]

    print(
        "\n================================"
    )

    print(
        "CLASSICAL BASELINE SUMMARY"
    )

    print(
        "================================"
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
        f"Mean IoU: "
        f"{np.mean(iou_scores):.4f}"
    )

    print(
        f"Median IoU: "
        f"{np.median(iou_scores):.4f}"
    )

    print(
        f"\nSaved to:\n{csv_path}"
    )


if __name__ == "__main__":
    main()