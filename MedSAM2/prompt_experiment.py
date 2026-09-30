import csv
import os
from glob import glob

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from skimage import measure

from sam2.build_sam import build_sam2_video_predictor_npz


# ============================================================
# SETTINGS
# ============================================================

CT_DIR = r"C:\Users\Nupur\Helios\data\output"
MASK_DIR = r"C:\Users\Nupur\Helios\data\output"

OUTPUT_DIR = r"C:\Users\Nupur\Helios\data\prompt_experiment"

CHECKPOINT = "checkpoints/MedSAM2_latest.pt"
CONFIG = "configs/sam2.1_hiera_t512.yaml"

MAX_PATIENTS = 5

# The only thing we are changing
BOX_MARGINS = [0, 5, 10, 20]


torch.set_float32_matmul_precision("high")
torch.manual_seed(2024)

if torch.cuda.is_available():
    torch.cuda.manual_seed(2024)

np.random.seed(2024)


# ============================================================
# LOAD
# ============================================================

def load_image(path):

    image = sitk.ReadImage(path)

    array = sitk.GetArrayFromImage(image)

    return image, array


# ============================================================
# OLD / BASELINE PREPROCESSING
# ============================================================

def preprocess_ct(ct):

    ct = np.asarray(
        ct,
        dtype=np.float32
    )

    # Our original baseline
    ct = np.clip(
        ct,
        -1000,
        400
    )

    # Convert to [0,255]
    ct = (
        (ct + 1000.0)
        / 1400.0
        * 255.0
    )

    ct = np.clip(
        ct,
        0,
        255
    )

    return ct.astype(np.uint8)


# ============================================================
# RESIZE CT
# ============================================================

def resize_ct(array, image_size=512):

    d, h, w = array.shape

    resized = np.zeros(
        (
            d,
            3,
            image_size,
            image_size
        ),
        dtype=np.float32
    )

    for i in range(d):

        img = Image.fromarray(
            array[i]
        )

        img = img.convert("RGB")

        img = img.resize(
            (
                image_size,
                image_size
            )
        )

        img = (
            np.asarray(img)
            .astype(np.float32)
            / 255.0
        )

        resized[i] = img.transpose(
            2,
            0,
            1
        )

    return resized


# ============================================================
# LARGEST TUMOR SLICE
# ============================================================

def find_key_slice(mask):

    areas = np.sum(
        mask > 0,
        axis=(1, 2)
    )

    if np.all(
        areas == 0
    ):
        return None

    return int(
        np.argmax(areas)
    )


# ============================================================
# BOUNDING BOX
# ============================================================

def compute_bbox(
    mask,
    key_slice,
    margin,
    height,
    width
):

    slice_mask = (
        mask[key_slice] > 0
    )

    ys, xs = np.where(
        slice_mask
    )

    if len(xs) == 0:

        return None

    x_min = max(
        0,
        int(xs.min()) - margin
    )

    y_min = max(
        0,
        int(ys.min()) - margin
    )

    x_max = min(
        width - 1,
        int(xs.max()) + margin
    )

    y_max = min(
        height - 1,
        int(ys.max()) + margin
    )

    return np.array(
        [
            x_min,
            y_min,
            x_max,
            y_max
        ],
        dtype=np.float32
    )


# ============================================================
# MEDSAM2 INFERENCE
# ============================================================

def run_inference(
    predictor,
    ct_data,
    key_slice,
    bbox
):

    img = resize_ct(
        ct_data,
        512
    )

    img = torch.from_numpy(
        img
    ).cuda()

    # ImageNet normalization
    mean = torch.tensor(
        (
            0.485,
            0.456,
            0.406
        ),
        dtype=torch.float32
    ).cuda()[:, None, None]

    std = torch.tensor(
        (
            0.229,
            0.224,
            0.225
        ),
        dtype=torch.float32
    ).cuda()[:, None, None]

    img = (
        img - mean
    ) / std

    height = ct_data.shape[1]
    width = ct_data.shape[2]

    segmentation = np.zeros(
        ct_data.shape,
        dtype=np.uint8
    )

    with torch.inference_mode():

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16
        ):

            # ------------------------------------------------
            # Forward propagation
            # ------------------------------------------------

            state = predictor.init_state(
                img,
                height,
                width
            )

            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=key_slice,
                obj_id=1,
                box=bbox
            )

            for (
                frame_idx,
                obj_ids,
                mask_logits
            ) in predictor.propagate_in_video(
                state
            ):

                mask = (
                    mask_logits[0] > 0
                ).cpu().numpy()[0]

                segmentation[
                    frame_idx
                ][mask] = 1

            predictor.reset_state(
                state
            )

            # ------------------------------------------------
            # Reverse propagation
            # ------------------------------------------------

            state = predictor.init_state(
                img,
                height,
                width
            )

            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=key_slice,
                obj_id=1,
                box=bbox
            )

            for (
                frame_idx,
                obj_ids,
                mask_logits
            ) in predictor.propagate_in_video(
                state,
                reverse=True
            ):

                mask = (
                    mask_logits[0] > 0
                ).cpu().numpy()[0]

                segmentation[
                    frame_idx
                ][mask] = 1

            predictor.reset_state(
                state
            )

    # Largest connected component
    if segmentation.max() > 0:

        labels = measure.label(
            segmentation
        )

        if labels.max() > 0:

            largest = (
                np.argmax(
                    np.bincount(
                        labels.flat
                    )[1:]
                ) + 1
            )

            segmentation = (
                labels == largest
            ).astype(
                np.uint8
            )

    return segmentation


# ============================================================
# METRICS
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

    gt_voxels = gt.sum()
    pred_voxels = pred.sum()

    union = np.logical_or(
        gt,
        pred
    ).sum()

    dice = (
        2.0 * intersection
        / (gt_voxels + pred_voxels)
        if gt_voxels + pred_voxels > 0
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


# ============================================================
# MAIN
# ============================================================

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
    )

    ct_files = ct_files[
        :MAX_PATIENTS
    ]

    print(
        f"Patients: {len(ct_files)}"
    )

    print(
        f"Box margins: {BOX_MARGINS}"
    )

    print(
        "\nLoading MedSAM2..."
    )

    predictor = (
        build_sam2_video_predictor_npz(
            CONFIG,
            CHECKPOINT
        )
    )

    results = []

    # ========================================================
    # PATIENT LOOP
    # ========================================================

    for patient_index, ct_path in enumerate(
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
            "\n================================"
        )

        print(
            f"[{patient_index}/{len(ct_files)}] "
            f"{patient}"
        )

        mask_path = os.path.join(
            MASK_DIR,
            f"{patient}_tumor_mask.nii.gz"
        )

        if not os.path.exists(
            mask_path
        ):

            print(
                "Missing mask. Skipping."
            )

            continue

        # Load
        _, ct = load_image(
            ct_path
        )

        _, gt = load_image(
            mask_path
        )

        if ct.shape != gt.shape:

            print(
                "Shape mismatch. Skipping."
            )

            continue

        if np.all(
            gt == 0
        ):

            print(
                "Empty mask. Skipping."
            )

            continue

        # Old baseline preprocessing
        ct_preprocessed = (
            preprocess_ct(ct)
        )

        key_slice = (
            find_key_slice(gt)
        )

        print(
            f"Key slice: {key_slice}"
        )

        height = gt.shape[1]
        width = gt.shape[2]

        # ====================================================
        # BOX MARGIN EXPERIMENT
        # ====================================================

        for margin in BOX_MARGINS:

            bbox = compute_bbox(
                gt,
                key_slice,
                margin,
                height,
                width
            )

            print(
                f"\n  Margin: "
                f"{margin}px"
            )

            print(
                f"  BBox: "
                f"{bbox.tolist()}"
            )

            prediction = run_inference(
                predictor,
                ct_preprocessed,
                key_slice,
                bbox
            )

            dice, iou = (
                calculate_metrics(
                    gt,
                    prediction
                )
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
                    margin,
                    dice,
                    iou
                ]
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    csv_path = os.path.join(
        OUTPUT_DIR,
        "prompt_experiment_results.csv"
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
                "box_margin_px",
                "dice",
                "iou"
            ]
        )

        writer.writerows(
            results
        )

    # ========================================================
    # SUMMARY BY MARGIN
    # ========================================================

    print(
        "\n\n================================"
    )

    print(
        "PROMPT EXPERIMENT SUMMARY"
    )

    print(
        "================================"
    )

    for margin in BOX_MARGINS:

        values = [
            r
            for r in results
            if r[1] == margin
        ]

        if not values:
            continue

        dice_scores = [
            r[2]
            for r in values
        ]

        iou_scores = [
            r[3]
            for r in values
        ]

        print(
            f"\nMargin: {margin}px"
        )

        print(
            f"  Mean Dice: "
            f"{np.mean(dice_scores):.4f}"
        )

        print(
            f"  Median Dice: "
            f"{np.median(dice_scores):.4f}"
        )

        print(
            f"  Mean IoU: "
            f"{np.mean(iou_scores):.4f}"
        )

        print(
            f"  Median IoU: "
            f"{np.median(iou_scores):.4f}"
        )

    print(
        f"\nCSV saved to:\n{csv_path}"
    )


if __name__ == "__main__":
    main()