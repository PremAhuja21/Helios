import os
import csv
from glob import glob

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from skimage import measure

from sam2.build_sam import build_sam2_video_predictor_npz


CT_DIR = r"C:\Users\Nupur\Helios\data\output"
MASK_DIR = r"C:\Users\Nupur\Helios\data\output"

OUTPUT_DIR = r"C:\Users\Nupur\Helios\data\paper_protocol"

CHECKPOINT = "checkpoints/MedSAM2_latest.pt"
CONFIG = "configs/sam2.1_hiera_t512.yaml"

MAX_PATIENTS = 5


torch.set_float32_matmul_precision("high")
torch.manual_seed(2024)

if torch.cuda.is_available():
    torch.cuda.manual_seed(2024)

np.random.seed(2024)


def load_nifti(path):

    image = sitk.ReadImage(path)

    data = sitk.GetArrayFromImage(image)

    return image, data


def preprocess_ct(ct):

    # Original baseline preprocessing
    ct = np.asarray(
        ct,
        dtype=np.float32
    )

    ct = np.clip(
        ct,
        -1000,
        400
    )

    ct = (
        (ct + 1000.0)
        / 1400.0
        * 255.0
    )

    return np.clip(
        ct,
        0,
        255
    ).astype(np.uint8)


def resize_volume(array, size=512):

    d, h, w = array.shape

    output = np.zeros(
        (
            d,
            3,
            size,
            size
        ),
        dtype=np.float32
    )

    for i in range(d):

        img = Image.fromarray(
            array[i]
        )

        img = img.convert("RGB")

        img = img.resize(
            (size, size)
        )

        img = (
            np.asarray(img)
            .astype(np.float32)
            / 255.0
        )

        output[i] = img.transpose(
            2,
            0,
            1
        )

    return output


def find_middle_tumor_slice(mask):

    z = np.where(
        np.any(
            mask > 0,
            axis=(1, 2)
        )
    )[0]

    if len(z) == 0:

        return None

    # Middle slice of the tumor's
    # occupied z-range
    return int(
        (z.min() + z.max()) // 2
    )


def compute_bbox(mask, slice_idx):

    ys, xs = np.where(
        mask[slice_idx] > 0
    )

    if len(xs) == 0:

        return None

    return np.array(
        [
            xs.min(),
            ys.min(),
            xs.max(),
            ys.max()
        ],
        dtype=np.float32
    )


def largest_cc(mask):

    labels = measure.label(mask)

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


def run_medsam2(
    predictor,
    ct,
    key_slice,
    bbox
):

    img = resize_volume(
        ct,
        512
    )

    img = torch.from_numpy(
        img
    ).cuda()

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

    h = ct.shape[1]
    w = ct.shape[2]

    prediction = np.zeros(
        ct.shape,
        dtype=np.uint8
    )

    with torch.inference_mode():

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16
        ):

            # Forward
            state = predictor.init_state(
                img,
                h,
                w
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
                logits
            ) in predictor.propagate_in_video(
                state
            ):

                mask = (
                    logits[0] > 0
                ).cpu().numpy()[0]

                prediction[
                    frame_idx
                ][mask] = 1

            predictor.reset_state(
                state
            )

            # Reverse
            state = predictor.init_state(
                img,
                h,
                w
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
                logits
            ) in predictor.propagate_in_video(
                state,
                reverse=True
            ):

                mask = (
                    logits[0] > 0
                ).cpu().numpy()[0]

                prediction[
                    frame_idx
                ][mask] = 1

            predictor.reset_state(
                state
            )

    return largest_cc(
        prediction
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
    )

    iou = (
        intersection / union
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

    print(
        f"Running paper-protocol "
        f"test on {len(ct_files)} patients."
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
            "\n================================"
        )

        print(
            f"[{i}/{len(ct_files)}] "
            f"{patient}"
        )

        mask_path = os.path.join(
            MASK_DIR,
            f"{patient}_tumor_mask.nii.gz"
        )

        _, ct = load_nifti(
            ct_path
        )

        _, gt = load_nifti(
            mask_path
        )

        ct = preprocess_ct(
            ct
        )

        key_slice = (
            find_middle_tumor_slice(
                gt
            )
        )

        bbox = compute_bbox(
            gt,
            key_slice
        )

        print(
            f"  Middle tumor slice: "
            f"{key_slice}"
        )

        print(
            f"  BBox: "
            f"{bbox.tolist()}"
        )

        prediction = run_medsam2(
            predictor,
            ct,
            key_slice,
            bbox
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
                key_slice,
                dice,
                iou
            ]
        )

    csv_path = os.path.join(
        OUTPUT_DIR,
        "paper_protocol_results.csv"
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
                "middle_tumor_slice",
                "dice",
                "iou"
            ]
        )

        writer.writerows(
            results
        )

    dice_scores = [
        r[2]
        for r in results
    ]

    iou_scores = [
        r[3]
        for r in results
    ]

    print(
        "\n================================"
    )

    print(
        "PAPER PROTOCOL SUMMARY"
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