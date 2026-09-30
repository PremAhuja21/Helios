import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

try:
    import pydicom_seg
    HAVE_PYDICOM_SEG = True
except ImportError:
    HAVE_PYDICOM_SEG = False

try:
    from rt_utils import RTStructBuilder
    HAVE_RT_UTILS = True
except ImportError:
    HAVE_RT_UTILS = False


def find_series_dirs(patient_dir: Path):
    result = {"CT": None, "RTSTRUCT": [], "SEG": []}
    ct_slice_count = -1

    for dirpath, _, filenames in os.walk(patient_dir):
        dcm_files = [f for f in filenames if f.lower().endswith(".dcm")]
        if not dcm_files:
            continue

        modality_map = {}
        for fname in dcm_files:
            fpath = os.path.join(dirpath, fname)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
            except Exception:
                continue
            modality = getattr(ds, "Modality", None)
            modality_map.setdefault(modality, []).append(fpath)

        if "CT" in modality_map:
            n = len(modality_map["CT"])
            if n > ct_slice_count:
                ct_slice_count = n
                result["CT"] = Path(dirpath)

        for fpath in modality_map.get("RTSTRUCT", []):
            result["RTSTRUCT"].append(Path(fpath))

        for fpath in modality_map.get("SEG", []):
            result["SEG"].append(Path(fpath))

    return result


def load_ct_volume(ct_dir: Path):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(ct_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {ct_dir}")

    best_files = []
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(str(ct_dir), sid)
        if len(files) > len(best_files):
            best_files = files

    reader.SetFileNames(best_files)
    ct_image = reader.Execute()
    return ct_image


def load_mask_from_seg(seg_paths, ct_image: sitk.Image):
    if not HAVE_PYDICOM_SEG:
        raise RuntimeError("pydicom-seg not installed (pip install pydicom-seg)")

    if isinstance(seg_paths, (str, Path)):
        seg_paths = [seg_paths]

    errors = []
    for seg_path in seg_paths:
        try:
            dcm = pydicom.dcmread(str(seg_path))
            reader = pydicom_seg.MultiClassReader()
            result = reader.read(dcm)

            seg_image = result.image
            seg_array = sitk.GetArrayFromImage(seg_image)
            tumor_mask = (seg_array > 0).astype(np.uint8)

            if tumor_mask.sum() == 0:
                errors.append(f"{seg_path.name}: mask is empty")
                continue

            mask_image = sitk.GetImageFromArray(tumor_mask)
            mask_image.CopyInformation(seg_image)

            if (mask_image.GetSize() != ct_image.GetSize()
                    or mask_image.GetSpacing() != ct_image.GetSpacing()):
                mask_image = sitk.Resample(
                    mask_image, ct_image, sitk.Transform(),
                    sitk.sitkNearestNeighbor, 0, mask_image.GetPixelID()
                )

            return mask_image, seg_path

        except Exception as e:
            errors.append(f"{seg_path.name}: {type(e).__name__}: {e}")
            continue

    raise RuntimeError(f"No usable SEG among {len(seg_paths)} file(s). Details: {errors}")


def _pick_gtv_roi(roi_names, preferred_roi_names=("GTV-1", "GTV1", "GTV")):
    for candidate in preferred_roi_names:
        if candidate in roi_names:
            return candidate
    matches = [r for r in roi_names if "gtv" in r.lower()]
    return matches[0] if matches else None


def load_mask_from_rtstruct(ct_dir: Path, rtstruct_paths, ct_image: sitk.Image):
    if not HAVE_RT_UTILS:
        raise RuntimeError("rt-utils not installed (pip install rt-utils)")

    if isinstance(rtstruct_paths, (str, Path)):
        rtstruct_paths = [rtstruct_paths]

    errors = []
    for rt_path in rtstruct_paths:
        try:
            rtstruct = RTStructBuilder.create_from(
                dicom_series_path=str(ct_dir),
                rt_struct_path=str(rt_path),
            )
            roi_names = rtstruct.get_roi_names()
            roi_to_use = _pick_gtv_roi(roi_names)
            if roi_to_use is None:
                errors.append(f"{rt_path.name}: no GTV-like ROI (found: {roi_names})")
                continue

            mask_3d = rtstruct.get_roi_mask_by_name(roi_to_use)
            mask_3d = np.transpose(mask_3d, (2, 0, 1)).astype(np.uint8)

            mask_image = sitk.GetImageFromArray(mask_3d)
            mask_image.CopyInformation(ct_image)

            return mask_image, rt_path, roi_to_use

        except Exception as e:
            errors.append(f"{rt_path.name}: {type(e).__name__}: {e}")
            continue

    raise RuntimeError(f"No usable RTSTRUCT among {len(rtstruct_paths)} file(s). Details: {errors}")


def convert_patient(patient_dir: Path, out_dir: Path, log_rows: list):
    patient_id = patient_dir.name

    try:
        series = find_series_dirs(patient_dir)

        if series["CT"] is None:
            log_rows.append([patient_id, "skipped", "no CT series found"])
            return

        ct_image = load_ct_volume(series["CT"])

        mask_image = None
        mask_detail = None

        if series["SEG"]:
            try:
                mask_image, seg_used = load_mask_from_seg(series["SEG"], ct_image)
                mask_detail = f"SEG file={seg_used.name}"
            except Exception:
                mask_image = None

        if mask_image is None and series["RTSTRUCT"]:
            try:
                mask_image, rt_used, roi_used = load_mask_from_rtstruct(
                    series["CT"], series["RTSTRUCT"], ct_image
                )
                mask_detail = f"RTSTRUCT file={rt_used.name} roi={roi_used}"
            except Exception as e:
                log_rows.append([patient_id, "skipped", f"RTSTRUCT parsing failed: {e}"])
                return

        if mask_image is None:
            log_rows.append([patient_id, "skipped", "no usable SEG or RTSTRUCT found"])
            return

        if mask_image.GetSize() != ct_image.GetSize():
            log_rows.append([patient_id, "error",
                              f"mask/CT size mismatch: {mask_image.GetSize()} vs {ct_image.GetSize()}"])
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(ct_image, str(out_dir / f"{patient_id}_ct.nii.gz"))
        sitk.WriteImage(mask_image, str(out_dir / f"{patient_id}_tumor_mask.nii.gz"))

        log_rows.append([patient_id, "success", mask_detail])

    except Exception as e:
        log_rows.append([patient_id, "error", f"{type(e).__name__}: {e}"])
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Batch-convert NSCLC-Radiomics DICOM data to NIfTI")
    parser.add_argument("--root", required=False,
                        default=r"C:\Users\Nupur\Helios\data\nsclc_radiomics",
                        help="Path to downloaded NSCLC-Radiomics root folder")
    parser.add_argument("--out", required=False,
                        default=r"C:\Users\Nupur\Helios\data\output",
                        help="Output folder for .nii.gz files + log")
    args = parser.parse_args()
    root = args.root
    out = args.out

    if not root:
        root = input("Enter path to downloaded NSCLC-Radiomics root folder: ").strip()
        while not root or not Path(root).exists():
            print("Path does not exist. Please enter a valid folder path.")
            root = input("Enter path to downloaded NSCLC-Radiomics root folder: ").strip()

    if not out:
        out = input("Enter output folder to write .nii.gz files (will be created if missing): ").strip()
        while not out:
            print("Please enter a valid output folder path.")
            out = input("Enter output folder to write .nii.gz files (will be created if missing): ").strip()

    root_dir = Path(root)
    out_dir = Path(out)

    if not HAVE_PYDICOM_SEG:
        print("[warning] pydicom-seg not installed — SEG objects will be skipped.")
    if not HAVE_RT_UTILS:
        print("[warning] rt-utils not installed — RTSTRUCT fallback unavailable.")

    patient_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    if not patient_dirs:
        print(f"No patient folders found under {root_dir}")
        sys.exit(1)

    print(f"Found {len(patient_dirs)} patient folders. Starting conversion...")

    log_rows = []
    for i, patient_dir in enumerate(patient_dirs, 1):
        print(f"[{i}/{len(patient_dirs)}] Processing {patient_dir.name} ...")
        convert_patient(patient_dir, out_dir, log_rows)

    log_path = out_dir / "conversion_log.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "status", "detail"])
        writer.writerows(log_rows)

    n_success = sum(1 for row in log_rows if row[1] == "success")
    n_skipped = sum(1 for row in log_rows if row[1] == "skipped")
    n_error = sum(1 for row in log_rows if row[1] == "error")

    print(f"\nDone. success={n_success}  skipped={n_skipped}  error={n_error}")
    print(f"Full log written to {log_path}")


if __name__ == "__main__":
    main()