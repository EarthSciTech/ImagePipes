"""
Code Description:
This script applies a 2D Region of Interest (ROI) mask, defined in an ImageJ .roi file, to TIFF stacks to set pixels within the ROI to zero. It:
1. Reads the ROI coordinates from an ImageJ .roi file, accounting for bounding box offsets.
2. Processes TIFF stacks in a specified folder, applying the ROI mask to each slice.
3. Saves the masked stacks in a new output folder, preserving the original data type and using zlib compression.
Key features:
- Safely handles ROI coordinates by clipping to image boundaries to prevent access violations.
- Supports both 2D and 3D TIFF stacks, treating 2D images as single-slice stacks.
- Maintains the original data type and applies the mask consistently across all slices.
Prerequisites:
- Input TIFF files must be readable by tifffile and represent single-band image data.
- The .roi file must be in ImageJ format with polygon coordinates.
Dependencies: numpy, tifffile, scikit-image (install via pip install numpy tifffile scikit-image).

Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, Mar 2025.
"""

import numpy as np  # Import NumPy for numerical operations and array handling
import tifffile  # Import tifffile for reading and writing TIFF files
import os  # Import os module for file and directory operations
from skimage.draw import polygon  # Import polygon function to generate mask from ROI coordinates

# Paths to the ROI and the folder containing TIFF stacks
roi_path = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\0-ROI\remove.roi"  # Path to the ImageJ ROI file defining the region to mask
tiff_folder = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\2-MaskTomograms\3-NonsolidMask\Sw1"  # Input folder containing TIFF stacks
output_folder = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\2-MaskTomograms\3-NonsolidMask\Sw1_ROI"  # Output folder for masked TIFF stacks
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn’t exist, no error if already present

# Function to load ImageJ ROI including bounding box offset
def read_imagej_roi(roi_file):  # Read coordinates from an ImageJ .roi file
    with open(roi_file, 'rb') as f:  # Open the ROI file in binary read mode
        data = f.read()  # Read all bytes from the file

    top = int.from_bytes(data[8:10], byteorder='big')  # Extract top offset (y) from bytes 8-9 (big-endian)
    left = int.from_bytes(data[10:12], byteorder='big')  # Extract left offset (x) from bytes 10-11 (big-endian)

    header_size = 64  # Define the size of the ROI header in bytes
    coords_offset = header_size + 4  # Calculate offset where coordinate data starts

    n = int.from_bytes(data[16:18], byteorder='big')  # Extract number of coordinates from bytes 16-17
    x_coords = np.frombuffer(data[coords_offset:coords_offset + 2*n], '>i2') + left  # Read x-coordinates and add left offset
    y_coords = np.frombuffer(data[coords_offset + 2*n:coords_offset + 4*n], '>i2') + top  # Read y-coordinates and add top offset

    return y_coords, x_coords  # Return y and x coordinates of the ROI polygon

# Safe polygon function to prevent access violation
def safe_polygon(y_coords, x_coords, shape):  # Generate polygon mask with clipped coordinates
    y_coords = np.clip(y_coords, 0, shape[-2] - 1)  # Clip y-coordinates to valid range [0, height-1]
    x_coords = np.clip(x_coords, 0, shape[-1] - 1)  # Clip x-coordinates to valid range [0, width-1]
    return polygon(y_coords, x_coords, shape=shape)  # Return row, column indices of pixels inside the polygon

# Load ROI coordinates with proper offsets
y_roi, x_roi = read_imagej_roi(roi_path)  # Read ROI coordinates from the .roi file

# Process each TIFF stack in the folder
for filename in os.listdir(tiff_folder):  # Iterate over files in the input folder
    if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):  # Check if file is a TIFF
        tiff_path = os.path.join(tiff_folder, filename)  # Construct full path to the TIFF file
        stack = tifffile.imread(tiff_path)  # Read the TIFF stack into a NumPy array
        modified_stack = stack.copy()  # Create a copy to avoid modifying the original data

        # Ensure stack is 3D
        if modified_stack.ndim == 2:  # If the stack is 2D (single slice)
            modified_stack = modified_stack[np.newaxis, :, :]  # Add a singleton dimension to make it 3D

        # Generate a 2D mask based on ROI (assuming ROI applies to each slice)
        mask2d = np.zeros(modified_stack.shape[-2:], dtype=bool)  # Create a 2D boolean mask (height, width)
        rr, cc = safe_polygon(y_roi, x_roi, shape=mask2d.shape)  # Get row, column indices for ROI polygon
        mask2d[rr, cc] = True  # Set pixels inside the ROI to True in the mask

        # Apply mask to each slice individually
        for i in range(modified_stack.shape[0]):  # Iterate over slices in the stack
            modified_stack[i][mask2d] = np.array(0, dtype=modified_stack.dtype)  # Set ROI pixels to 0 in the slice

        # Save modified stack
        output_path = os.path.join(output_folder, f'{filename}')  # Construct output file path
        tifffile.imwrite(output_path, modified_stack.squeeze(), dtype=stack.dtype, compression='zlib')  # Save masked stack with original dtype and zlib compression
        print(f'Saved masked stack: {output_path}')  # Log the saved file path
        
# Print a message to indicate that the entire processing is complete
print("✅ Denoising complete.")  # Print a final message when all processing is done