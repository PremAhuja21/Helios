import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to

# ==========================================
# FOLDERS
# ==========================================
ct_folder = r"D:\Helios\data\output"
prediction_folder = r"D:\Helios\data\batch_predictions"
output_folder = r"D:\Helios\output\ct_tumor_overlay"

os.makedirs(output_folder, exist_ok=True)

# Find all MedSAM2 prediction files
prediction_files = sorted(
    glob.glob(
        os.path.join(prediction_folder, "*_prediction.nii.gz")
    )
)

print(f"Found {len(prediction_files)} prediction files.\n")

successful = 0
failed = 0

# ==========================================
# PROCESS EACH PATIENT
# ==========================================
for prediction_path in prediction_files:

    patient_name = os.path.basename(
        prediction_path
    ).replace("_prediction.nii.gz", "")

    ct_path = os.path.join(
        ct_folder,
        f"{patient_name}_ct.nii.gz"
    )

    print("=" * 55)
    print(f"Processing: {patient_name}")

    try:
        # Check CT file exists
        if not os.path.exists(ct_path):
            print("CT file not found. Skipping.")
            failed += 1
            continue

        # Load CT and prediction
        ct_img = nib.load(ct_path)
        prediction_img = nib.load(prediction_path)

        print("CT shape:", ct_img.shape)
        print("Prediction shape:", prediction_img.shape)

        # ==========================================
        # ALIGN PREDICTION TO CT SPACE
        # order=0 preserves binary mask values
        # ==========================================
        prediction_aligned_img = resample_from_to(
            prediction_img,
            ct_img,
            order=0
        )

        ct = ct_img.get_fdata()
        prediction = prediction_aligned_img.get_fdata()

        # Ensure binary mask
        prediction = (prediction > 0.5).astype(np.uint8)

        print("Aligned prediction shape:", prediction.shape)

        # ==========================================
        # FIND SLICE WITH MOST TUMOR
        # ==========================================
        tumor_per_slice = np.sum(
            prediction > 0,
            axis=(0, 1)
        )

        slice_number = np.argmax(tumor_per_slice)
        tumor_pixels = tumor_per_slice[slice_number]

        if tumor_pixels == 0:
            print("No tumor found. Skipping.")
            failed += 1
            continue

        print("Best tumor slice:", slice_number)
        print("Tumor pixels:", tumor_pixels)

        # Get slices
        ct_slice = ct[:, :, slice_number]
        tumor_slice = prediction[:, :, slice_number]

        # ==========================================
        # CREATE VISUALIZATION
        # ==========================================
        plt.figure(figsize=(10, 10))

        plt.imshow(
            ct_slice.T,
            cmap="gray",
            origin="lower"
        )

        plt.contour(
            tumor_slice.T,
            levels=[0.5],
            linewidths=2
        )

        plt.title(
            f"{patient_name} | CT + MedSAM2 Predicted Tumor\n"
            f"Slice {slice_number}"
        )

        plt.axis("off")

        # ==========================================
        # SAVE IMAGE
        # ==========================================
        output_file = os.path.join(
            output_folder,
            f"{patient_name}_ct_tumor_overlay.png"
        )

        plt.savefig(
            output_file,
            bbox_inches="tight",
            dpi=200
        )

        # Close figure so 37 windows don't open
        plt.close()

        print("Saved:", output_file)
        successful += 1

    except Exception as e:
        print(f"ERROR: {e}")
        failed += 1


# ==========================================
# FINAL SUMMARY
# ==========================================
print("\n" + "=" * 55)
print("CT + TUMOR OVERLAY BATCH COMPLETE")
print("Successfully processed:", successful)
print("Failed / skipped:", failed)
print("Output folder:", output_folder)
print("=" * 55)