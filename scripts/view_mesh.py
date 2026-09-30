import os
import pyvista as pv

# ==============================
# ASK FOR PATIENT NUMBER
# ==============================
patient_number = input(
    "Enter patient number (example: 1, 35, 100, 422): "
)

# Convert number to format: LUNG1-001, LUNG1-100, etc.
patient_name = f"LUNG1-{int(patient_number):03d}"

# Path of the saved 3D model
file_path = (
    f"D:\\Helios\\output\\marching_cubes\\"
    f"{patient_name}_marching_cubes.ply"
)

# ==============================
# CHECK IF MODEL EXISTS
# ==============================
if not os.path.exists(file_path):
    print(f"\nModel not found for {patient_name}")
else:
    print(f"\nLoading 3D model for {patient_name}...")

    # Load the saved 3D tumor model
    mesh = pv.read(file_path)

    print(mesh)

    # Open interactive 3D viewer
    plotter = pv.Plotter()
    plotter.add_mesh(mesh)
    plotter.add_axes()
    plotter.show()