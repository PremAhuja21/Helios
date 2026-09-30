import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

try:
    from lungmask import LMInferer
    HAVE_LUNGMASK = True
except ImportError:
    HAVE_LUNGMASK = False


def run_lungmask(ct_path: Path, inferer) -> sitk.Image:
    ct_image = sitk.ReadImage(str(ct_path))
    lung_array = inferer.apply(ct_image)
    lung_image = sitk.GetImageFromArray(lung_array.astype(np.uint8))
    lung_image.CopyInformation(ct_image)
    return lung_image


def combine_masks(lung_image: sitk.Image, tumor_image: sitk.Image) -> sitk.Image:
    lung_array = sitk.GetArrayFromImage(lung_image).astype(np.uint8)
    tumor_array = sitk.GetArrayFromImage(tumor_image).astype(np.uint8)

    if lung_array.shape != tumor_array.shape:
        raise RuntimeError(f"Shape mismatch: lung {lung_array.shape} vs tumor {tumor_array.shape}")

    combined = lung_array.copy()
    combined[tumor_array > 0] = 3

    combined_image = sitk.GetImageFromArray(combined)
    combined_image.CopyInformation(lung_image)
    return combined_image


def process_patient(patient_id, ct_path, tumor_path, out_dir, inferer, log_rows):
    try:
        if not tumor_path.exists():
            log_rows.append([patient_id, "skipped", "no tumor mask file found"])
            return

        lung_image = run_lungmask(ct_path, inferer)
        tumor_image = sitk.ReadImage(str(tumor_path))

        combined_image = combine_masks(lung_image, tumor_image)

        out_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(lung_image, str(out_dir / f"{patient_id}_lung_mask.nii.gz"))
        sitk.WriteImage(combined_image, str(out_dir / f"{patient_id}_combined_label.nii.gz"))

        log_rows.append([patient_id, "success", ""])

    except Exception as e:
        log_rows.append([patient_id, "error", f"{type(e).__name__}: {e}"])


def main():
    parser = argparse.ArgumentParser(description="Run lungmask and build combined lung+tumor label maps")
    parser.add_argument("--in", dest="in_dir", required=False, help="Folder with *_ct.nii.gz and *_tumor_mask.nii.gz")
    parser.add_argument("--out", dest="out_dir", required=False, help="Output folder")
    parser.add_argument("--model", default="R231", help="lungmask model variant (default R231)")
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir

    if not in_dir:
        in_dir = input("Enter folder containing *_ct.nii.gz and *_tumor_mask.nii.gz: ").strip()
    if not out_dir:
        out_dir = input("Enter output folder for lung masks + combined labels: ").strip()

    if not HAVE_LUNGMASK:
        print("lungmask not installed. Install with: pip install lungmask")
        sys.exit(1)

    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    ct_files = sorted(in_dir.glob("*_ct.nii.gz"))
    if not ct_files:
        print(f"No *_ct.nii.gz files found in {in_dir}")
        sys.exit(1)

    print(f"Found {len(ct_files)} CT files. Loading lungmask model ({args.model})...")
    inferer = LMInferer(modelname=args.model)

    log_rows = []
    for i, ct_path in enumerate(ct_files, 1):
        patient_id = ct_path.name.replace("_ct.nii.gz", "")
        tumor_path = in_dir / f"{patient_id}_tumor_mask.nii.gz"

        print(f"[{i}/{len(ct_files)}] {patient_id} ...")
        process_patient(patient_id, ct_path, tumor_path, out_dir, inferer, log_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "lungmask_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "status", "detail"])
        writer.writerows(log_rows)

    n_success = sum(1 for row in log_rows if row[1] == "success")
    n_skipped = sum(1 for row in log_rows if row[1] == "skipped")
    n_error = sum(1 for row in log_rows if row[1] == "error")

    print(f"\nDone. success={n_success}  skipped={n_skipped}  error={n_error}")
    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()