import argparse
import csv
import logging
import os
from glob import glob

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor_npz
from skimage import measure


# ============================================================
# SETTINGS
# ============================================================

torch.set_float32_matmul_precision("high")
torch.manual_seed(2024)

if torch.cuda.is_available():
    torch.cuda.manual_seed(2024)

np.random.seed(2024)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)


# ============================================================
# CONNECTED COMPONENT
# ============================================================

def get_largest_cc(segmentation):
    """
    Keep only the largest connected component
    from a binary 3D segmentation.
    """

    labels = measure.label(segmentation)

    if labels.max() == 0:
        return segmentation.astype(np.uint8)

    largest = labels == (
        np.argmax(
            np.bincount(labels.flat)[1:]
        ) + 1
    )

    return largest.astype(np.uint8)


# ============================================================
# IMAGE RESIZING
# ============================================================

def resize_grayscale_to_rgb_and_resize(
    array,
    image_size
):
    """
    Convert 3D grayscale CT volume to RGB
    and resize each slice to image_size x image_size.

    Input:
        (depth, height, width)

    Output:
        (depth, 3, image_size, image_size)
    """

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

        # Convert normalized [0,1] image
        # to uint8 for PIL resizing
        slice_uint8 = np.clip(
            array[i] * 255.0,
            0,
            255
        ).astype(np.uint8)

        img = Image.fromarray(
            slice_uint8
        )

        img = img.convert("RGB")

        img = img.resize(
            (image_size, image_size)
        )

        img_array = (
            np.asarray(img)
            .astype(np.float32)
            / 255.0
        )

        resized[i] = img_array.transpose(
            2,
            0,
            1
        )

    return resized


# ============================================================
# LOAD DATA
# ============================================================

def load_ct(path):

    image = sitk.ReadImage(path)

    data = sitk.GetArrayFromImage(
        image
    )

    return image, data


def load_mask(path):

    image = sitk.ReadImage(path)

    data = sitk.GetArrayFromImage(
        image
    )

    return image, data


# ============================================================
# RESAMPLING
# ============================================================

def resample_to_spacing(
    image,
    target_spacing,
    is_mask=False
):
    """
    Resample image to target physical spacing.

    CT:
        Linear interpolation.

    Mask:
        Nearest-neighbor interpolation.
    """

    original_spacing = image.GetSpacing()

    original_size = image.GetSize()

    new_size = [
        int(
            round(
                original_size[i]
                * original_spacing[i]
                / target_spacing[i]
            )
        )
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()

    resampler.SetOutputSpacing(
        target_spacing
    )

    resampler.SetSize(
        new_size
    )

    resampler.SetOutputDirection(
        image.GetDirection()
    )

    resampler.SetOutputOrigin(
        image.GetOrigin()
    )

    resampler.SetTransform(
        sitk.Transform()
    )

    resampler.SetDefaultPixelValue(
        0
    )

    if is_mask:

        resampler.SetInterpolator(
            sitk.sitkNearestNeighbor
        )

    else:

        resampler.SetInterpolator(
            sitk.sitkLinear
        )

    return resampler.Execute(
        image
    )


# ============================================================
# CT PREPROCESSING
# ============================================================

def preprocess_ct(ct_image):
    """
    CT preprocessing:

    1. Clip HU to [-1000, 1000]
    2. Resample to (1, 1, 3) mm
    3. Normalize to [0, 1]

    Returns:
        resampled CT SimpleITK image
        normalized CT NumPy array
    """

    target_spacing = (
        1.0,
        1.0,
        3.0
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    ct_array = (
        sitk.GetArrayFromImage(
            ct_image
        )
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # HU clipping
    # --------------------------------------------------------

    ct_array = np.clip(
        ct_array,
        -1000.0,
        1000.0
    )

    # --------------------------------------------------------
    # Convert back to SimpleITK
    # --------------------------------------------------------

    clipped_image = (
        sitk.GetImageFromArray(
            ct_array
        )
    )

    clipped_image.CopyInformation(
        ct_image
    )

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    resampled_image = (
        resample_to_spacing(
            clipped_image,
            target_spacing,
            is_mask=False
        )
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    ct = (
        sitk.GetArrayFromImage(
            resampled_image
        )
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # Normalize [-1000,1000] -> [0,1]
    # --------------------------------------------------------

    ct = (
        ct + 1000.0
    ) / 2000.0

    ct = np.clip(
        ct,
        0.0,
        1.0
    )

    return (
        resampled_image,
        ct
    )


# ============================================================
# FIND LARGEST TUMOR SLICE
# ============================================================

def find_key_slice(mask):

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
# BOUNDING BOX
# ============================================================

def compute_bbox(
    mask,
    key_slice
):

    if key_slice is None:
        return None

    slice_mask = (
        mask[key_slice] > 0
    )

    ys, xs = np.where(
        slice_mask
    )

    if len(xs) == 0:

        return None

    bbox = np.array(
        [
            xs.min(),
            ys.min(),
            xs.max(),
            ys.max()
        ],
        dtype=np.float32
    )

    return bbox


# ============================================================
# MEDSAM2 INFERENCE
# ============================================================

def run_inference(
    predictor,
    ct_data,
    key_slice,
    bbox
):

    # --------------------------------------------------------
    # Resize volume to 512 x 512
    # --------------------------------------------------------

    img = (
        resize_grayscale_to_rgb_and_resize(
            ct_data,
            512
        )
    )

    # --------------------------------------------------------
    # Convert to torch
    #
    # Image is already normalized to [0,1].
    # DO NOT divide by 255 again.
    # --------------------------------------------------------

    img = torch.from_numpy(
        img
    ).cuda()

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Original spatial dimensions
    # --------------------------------------------------------

    height = ct_data.shape[1]

    width = ct_data.shape[2]

    # --------------------------------------------------------
    # Output segmentation
    # --------------------------------------------------------

    segmentation = np.zeros(
        ct_data.shape,
        dtype=np.uint8
    )

    # ========================================================
    # MEDSAM2
    # ========================================================

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

    # --------------------------------------------------------
    # Largest connected component
    # --------------------------------------------------------

    if segmentation.max() > 0:

        segmentation = (
            get_largest_cc(
                segmentation
            )
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

    gt_voxels = int(
        gt.sum()
    )

    pred_voxels = int(
        pred.sum()
    )

    intersection = int(
        np.logical_and(
            gt,
            pred
        ).sum()
    )

    union = int(
        np.logical_or(
            gt,
            pred
        ).sum()
    )

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    if (
        gt_voxels
        + pred_voxels
        > 0
    ):

        dice = (
            2.0
            * intersection
            / (
                gt_voxels
                + pred_voxels
            )
        )

    else:

        dice = 1.0

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    if union > 0:

        iou = (
            intersection
            / union
        )

    else:

        iou = 1.0

    return (
        dice,
        iou,
        gt_voxels,
        pred_voxels
    )


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(
    reference_image,
    prediction,
    path
):

    output = (
        sitk.GetImageFromArray(
            prediction.astype(
                np.uint8
            )
        )
    )

    output.CopyInformation(
        reference_image
    )

    sitk.WriteImage(
        output,
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "MedSAM2 inference on "
            "NSCLC-Radiomics"
        )
    )

    parser.add_argument(
        "--ct_dir",
        required=True
    )

    parser.add_argument(
        "--mask_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/"
            "MedSAM2_latest.pt"
        )
    )

    parser.add_argument(
        "--cfg",
        default=(
            "configs/"
            "sam2.1_hiera_t512.yaml"
        )
    )

    parser.add_argument(
        "--max_patients",
        type=int,
        default=None
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find CT volumes
    # --------------------------------------------------------

    ct_files = sorted(
        glob(
            os.path.join(
                args.ct_dir,
                "*_ct.nii.gz"
            )
        )
    )

    if args.max_patients is not None:

        ct_files = (
            ct_files[
                :args.max_patients
            ]
        )

    print(
        f"Found {len(ct_files)} "
        "CT volumes to process."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "Loading MedSAM2 model..."
    )

    predictor = (
        build_sam2_video_predictor_npz(
            args.cfg,
            args.checkpoint
        )
    )

    results = []

    # ========================================================
    # PATIENT LOOP
    # ========================================================

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
            f"Processing {patient}"
        )

        mask_path = os.path.join(
            args.mask_dir,
            f"{patient}_tumor_mask.nii.gz"
        )

        if not os.path.exists(
            mask_path
        ):

            print(
                "  Missing ground-truth "
                "mask. Skipping."
            )

            continue

        try:

            # ------------------------------------------------
            # Load original CT
            # ------------------------------------------------

            ct_image, ct_data = (
                load_ct(
                    ct_path
                )
            )

            # ------------------------------------------------
            # Load original mask
            # ------------------------------------------------

            mask_image, gt_mask = (
                load_mask(
                    mask_path
                )
            )

            # ------------------------------------------------
            # Preprocess CT
            #
            # [-1000,1000] HU
            # -> 1x1x3 mm
            # -> [0,1]
            # ------------------------------------------------

            (
                ct_resampled_image,
                ct_preprocessed
            ) = preprocess_ct(
                ct_image
            )

            # ------------------------------------------------
            # Resample GT mask
            #
            # SAME grid as CT
            # Nearest neighbor
            # ------------------------------------------------

            mask_resampled_image = (
                resample_to_spacing(
                    mask_image,
                    (
                        1.0,
                        1.0,
                        3.0
                    ),
                    is_mask=True
                )
            )

            gt_mask = (
                sitk.GetArrayFromImage(
                    mask_resampled_image
                )
            )

            gt_mask = (
                gt_mask > 0
            ).astype(
                np.uint8
            )

            # ------------------------------------------------
            # Verify shapes
            # ------------------------------------------------

            if (
                ct_preprocessed.shape
                != gt_mask.shape
            ):

                print(
                    "  Shape mismatch "
                    "after resampling:"
                )

                print(
                    "  CT:",
                    ct_preprocessed.shape
                )

                print(
                    "  GT:",
                    gt_mask.shape
                )

                continue

            # ------------------------------------------------
            # Empty mask check
            # ------------------------------------------------

            if np.all(
                gt_mask == 0
            ):

                print(
                    "  Empty ground-truth "
                    "mask. Skipping."
                )

                continue

            # ------------------------------------------------
            # Find largest tumor slice
            # ------------------------------------------------

            key_slice = (
                find_key_slice(
                    gt_mask
                )
            )

            if key_slice is None:

                print(
                    "  Could not find "
                    "tumor slice."
                )

                continue

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            bbox = (
                compute_bbox(
                    gt_mask,
                    key_slice
                )
            )

            if bbox is None:

                print(
                    "  Could not compute "
                    "bounding box."
                )

                continue

            print(
                f"  CT shape after "
                f"resampling: "
                f"{ct_preprocessed.shape}"
            )

            print(
                f"  Key slice: "
                f"{key_slice}"
            )

            print(
                f"  Bounding box: "
                f"{bbox.tolist()}"
            )

            # ------------------------------------------------
            # Run MedSAM2
            # ------------------------------------------------

            prediction = (
                run_inference(
                    predictor,
                    ct_preprocessed,
                    key_slice,
                    bbox
                )
            )

            # ------------------------------------------------
            # Save prediction
            #
            # Save using RESAMPLED CT metadata
            # ------------------------------------------------

            pred_path = os.path.join(
                args.output_dir,
                f"{patient}_prediction.nii.gz"
            )

            save_prediction(
                ct_resampled_image,
                prediction,
                pred_path
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            (
                dice,
                iou,
                gt_voxels,
                pred_voxels
            ) = calculate_metrics(
                gt_mask,
                prediction
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

            results.append(
                (
                    patient,
                    key_slice,
                    bbox.tolist(),
                    dice,
                    iou,
                    gt_voxels,
                    pred_voxels,
                    "success"
                )
            )

        except Exception as e:

            print(
                f"  ERROR: "
                f"{type(e).__name__}: "
                f"{e}"
            )

            results.append(
                (
                    patient,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    f"error: {e}"
                )
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = os.path.join(
        args.output_dir,
        "results.csv"
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
                "key_slice",
                "bbox",
                "dice",
                "iou",
                "gt_voxels",
                "pred_voxels",
                "status"
            ]
        )

        writer.writerows(
            results
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    successful = [
        r
        for r in results
        if r[7] == "success"
    ]

    if successful:

        dice_scores = [
            r[3]
            for r in successful
        ]

        iou_scores = [
            r[4]
            for r in successful
        ]

        print(
            "\n=============================="
        )

        print(
            "BATCH SUMMARY"
        )

        print(
            "=============================="
        )

        print(
            f"Successful: "
            f"{len(successful)}"
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
        f"\nResults saved to: "
        f"{csv_path}"
    )


if __name__ == "__main__":
    main()