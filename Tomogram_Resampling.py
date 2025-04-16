"""
Code Description:
 This script resamples a 3D image composed of sequential 2D TIFF slices stored in a folder.
 The original dataset has isotropic voxel resolution (e.g., 4.61 µm in X, Y, and Z) and consists of 
 multiple 2D TIFF files representing Z slices (e.g., 1350 files for 1350 slices). 
 The script performs the following steps:
 
 1. Loads all 2D TIFF files, sorts them, and stacks them into a single 3D NumPy array.
 2. Applies 3D downsampling using scipy.ndimage.zoom with nearest-neighbor interpolation to preserve binary mask values.
 3. Saves the resampled 3D image back into a new folder, slice-by-slice, as 2D TIFF files.
 
 Key features:
 - Maintains binary mask integrity using nearest-neighbor interpolation (order=0).
 - Applies isotropic scaling to all dimensions (Z, Y, X) based on voxel size ratio.
 - Outputs each resampled slice as a separate TIFF image using uint8 format for compatibility.
 - Creates an output subdirectory named "resampled" inside the input folder.

 Prerequisites:
 - Input TIFF files must be 2D slices ordered in filenames.
 - All slices must have the same shape (X, Y) and represent an isotropic 3D image when stacked.

 Dependencies: os, tifffile, numpy, scipy
 Install via: pip install tifffile numpy scipy

 Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, April 2025.
"""

import os  # For file and folder operations
import tifffile as tiff  # For reading and writing TIFF images
import numpy as np  # For array and numerical operations
from scipy.ndimage import zoom  # For resampling via interpolation

# Define the input directory containing 2D TIFF slice files
input_dir = r"Z:\users\ream\2-AnalysedData\3-InertiaEffectData\2-MaskTomograms\5-Resampled\4_61um"

# Define the output directory for resampled files (subfolder of input)
output_dir = os.path.join(input_dir, "resampled")
os.makedirs(output_dir, exist_ok=True)  # Create output folder if it doesn't exist

# Define the current and desired isotropic voxel sizes in micrometers
current_voxel_size = 4.61  # Current voxel size in µm (Z, Y, X)
target_voxel_size = 20   # Target voxel size in µm (Z, Y, X)

# Calculate the zoom factor (scaling) for resampling
zoom_factor = current_voxel_size / target_voxel_size  # e.g., 0.461
zoom_factors = (zoom_factor, zoom_factor, zoom_factor)  # Apply to (Z, Y, X)

# Print one-time message indicating processing has started
print(f"Processing started for folder: {os.path.basename(input_dir)}")

# Get a sorted list of 2D TIFF files (assumes order matches slice index)
tif_files = sorted([
    f for f in os.listdir(input_dir) 
    if f.endswith(".tif") or f.endswith(".tiff")
])

# Initialize an empty list to store the 2D slices
stack = []

# Loop through each file, read it as a 2D array, and append to the stack
for fname in tif_files:
    img2d = tiff.imread(os.path.join(input_dir, fname))  # Read 2D slice
    stack.append(img2d)  # Append to list

# Convert the list of 2D slices into a 3D NumPy array (Z, Y, X)
img3d = np.stack(stack, axis=0)

# Print original shape for verification
original_shape = img3d.shape  # Shape = (Z, Y, X)
print(f"Original 3D shape: {original_shape}")

# Apply 3D resampling using nearest-neighbor to preserve binary values
resampled_img = zoom(img3d, zoom=zoom_factors, order=0)

# Print resampled shape for verification
resampled_shape = resampled_img.shape
print(f"Resampled 3D shape: {resampled_shape}")

# Save each 2D slice of the resampled 3D image as a separate TIFF file
for i in range(resampled_shape[0]):  # Iterate over Z slices
    slice_2d = resampled_img[i, :, :]  # Extract the i-th slice
    filename = f"slice{i:04d}.tif"  # Format the output filename (e.g., slice_0001.tif)
    tiff.imwrite(os.path.join(output_dir, filename), slice_2d.astype(np.uint8))  # Save slice as TIFF (0/1 values)

# Print final message upon completion
print(f"Finished processing all files in: {os.path.basename(input_dir)}")

print("✅ Processing complete.")