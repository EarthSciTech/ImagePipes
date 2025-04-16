"""
 Code Description:
 This script processes TIFF stacks in a main directory and its subdirectories to generate 3D surface meshes.
 It:
 1. Checks the main directory and all subdirectories for TIFF files.
 2. For each folder with TIFF files, loads them into a 3D volume, converts to a binary mask, and creates a surface mesh using marching cubes.
 3. Saves the resulting meshes as PLY file in a '3DMesh' subfolder within each processed folder.
 Key features:
 - Recursively processes folder hierarchies using os.walk.
 - Uses nearest-neighbor thresholding for binary mask conversion and preserves surface normals in the mesh.
 - Skips empty folders and avoids redundant processing of the main directory.
 Prerequisites:
 - Input TIFF files must be 2D slices of a 3D binary mask (e.g., 0 and 255 values).
 Dependencies: os, numpy, imageio, scikit-image (skimage), trimesh (install via pip install imageio scikit-image trimesh).

Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, April 2025.
"""

import os  # Import os module for file and directory operations
import numpy as np  # Import numpy for numerical array operations
import imageio.v3 as iio  # Import imageio.v3 for reading image files, aliased as iio
from skimage import measure  # Import measure module from skimage for marching cubes algorithm
import trimesh  # Import trimesh for creating and exporting 3D mesh objects

# Set the main directory
main_dir = r"Z:\users\ream\2-AnalysedData\3-InertiaEffectData\2-MaskTomograms\5-Resampled\20um"  # Define the main directory path to process

def process_tiff_stack(folder_path):  # Define a function to process a TIFF stack in a given folder
    """Processes TIFF stack in a given folder and saves PLY in a '3DMesh' subfolder."""  # Docstring describing the function’s purpose
    tif_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.tif')])  # Create a sorted list of TIFF files (case-insensitive)
    if not tif_files:  # Check if there are no TIFF files in the folder
        print(f"No TIFF files in: {folder_path}")  # Print a message if no TIFF files are found
        return  # Exit the function if no TIFF files exist

    print(f"Processing TIFF stack in: {folder_path}")  # Print a message indicating processing has started for this folder
    tif_paths = [os.path.join(folder_path, f) for f in tif_files]  # Generate full file paths for each TIFF file
    volume = np.stack([iio.imread(f) for f in tif_paths], axis=0)  # Load TIFF files and stack them into a 3D NumPy array
    binary_volume = (volume > 0).astype(np.uint8)  # Convert the volume to a binary mask (values > 0 become 1, others 0) as uint8
    
    # Generate surface mesh
    spacing = (1350/311, 910/210, 910/210) # spacing is used to scale up or scale down the voxel size isotropically in z, y, and x directions
    verts, faces, normals, _ = measure.marching_cubes(binary_volume, level=0.5, spacing=spacing)  # Use marching cubes to extract vertices, faces, and normals; level=0.5 is the isosurface threshold; 
    verts = verts[:, [2, 1, 0]] # Reorder vertices from (Z, Y, X) to (X, Y, Z)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)  # Create a Trimesh object with the extracted vertices, faces, and normals
    
    # Output folder
    output_folder = os.path.join(folder_path, "3DMesh")  # Define the output folder path as a '3DMesh' subdirectory
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn’t already exist, no error if it does
    mesh.export(os.path.join(output_folder, "3Dmesh.ply"))  # Export the mesh as a PLY file in the output folder

    print(f"Mesh saved in: {output_folder}")  # Print a confirmation message with the output folder path

# First, check if main_dir itself contains TIFF files
main_dir_tiffs = [f for f in os.listdir(main_dir) if f.lower().endswith('.tif')]  # Check for TIFF files directly in the main directory
if main_dir_tiffs:  # If TIFF files are found in the main directory
    process_tiff_stack(main_dir)  # Process the main directory as a TIFF stack

# Then, process subdirectories
for root, dirs, _ in os.walk(main_dir):  # Use os.walk to recursively traverse the main directory and its subdirectories
    if root == main_dir:  # Check if the current root is the main directory
        continue  # Skip processing the main directory here to avoid redundancy (already checked above)
    process_tiff_stack(root)  # Process the current subdirectory by calling the function