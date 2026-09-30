import os
import glob
import numpy as np
import nibabel as nib
import pyvista as pv
from skimage.measure import marching_cubes

# ==============================
# FOLDERS
# ==============================
input_folder = r"D:\Helios\data\batch_predictions"
output_folder = r"D:\Helios\output\marching_cubes"

os.makedirs(output_folder, exist_ok=True)

# Find all prediction files
prediction_files = sorted(
    glob.glob(os.path.join(input_folder, "*_prediction.nii.gz"))
)

print(f"Found {len(prediction_files)} prediction files.\n")

# ==============================
# PROCESS EACH PATIENT
# ==============================
successful = 0
failed = 0

for file_path in prediction_files:
    patient_name = os.path.basename(file_path).replace("_prediction.nii.gz", "")

    print("=" * 50)
    print(f"Processing: {patient_name}")

    try:
        # Load NIfTI
        img = nib.load(file_path)
        volume = img.get_fdata()

        # Get voxel spacing
        spacing = img.header.get_zooms()[:3]

        print("Shape:", volume.shape)
        print("Unique values:", np.unique(volume))
        print("Voxel spacing:", spacing)

        # Check whether tumor exists
        if not np.any(volume > 0):
            print("WARNING: No tumor found. Skipping.")
            failed += 1
            continue

        # ==============================
        # MARCHING CUBES
        # ==============================
        vertices, faces, normals, values = marching_cubes(
            volume,
            level=0.5,
            spacing=spacing
        )

        print("Vertices:", len(vertices))
        print("Faces:", len(faces))

        # ==============================
        # CREATE MESH
        # ==============================
        pv_faces = np.hstack([
            np.full((len(faces), 1), 3),
            faces
        ]).astype(np.int64)

        mesh = pv.PolyData(vertices, pv_faces)

        # ==============================
        # SAVE MESH
        # ==============================
        output_file = os.path.join(
            output_folder,
            f"{patient_name}_marching_cubes.ply"
        )

        mesh.save(output_file)

        print("Saved:", output_file)
        successful += 1

    except Exception as e:
        print(f"ERROR processing {patient_name}: {e}")
        failed += 1


# ==============================
# FINAL SUMMARY
# ==============================
print("\n" + "=" * 50)
print("BATCH PROCESSING COMPLETE")
print("Successfully processed:", successful)
print("Failed / skipped:", failed)
print("Output folder:", output_folder)
print("=" * 50)