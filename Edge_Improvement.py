"""
Code Description:
This script applies binary erosion to 3D TIFF stacks to improve edges by removing boundary voxels, preserving the original data type. It:
1. Recursively traverses an input directory to process TIFF stacks in subfolders.
2. Applies 3D binary erosion to the stack (with padding) and 2D erosion to the first and last slices for refined edge handling.
3. Saves the processed stacks in a mirrored output folder structure, maintaining the original data type.
Key features:
- Uses a 3x3x3 structuring element for 3D erosion and a 3x3 element for boundary slices to ensure consistent edge improvement.
- Handles subfolder structures automatically, creating corresponding output directories.
- Preserves zero-valued regions and applies erosion only to non-zero voxels.
Prerequisites:
- Input TIFF files must be readable by imageio and represent single-band image data.
Dependencies: os, imageio, numpy, scipy (install via pip install imageio numpy scipy).

Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, Feb 2025.
"""

import os  # Import os module for file and directory operations
import imageio.v2 as imageio  # Import imageio.v2 for reading and writing TIFF files
import numpy as np  # Import NumPy for numerical operations and array handling
from scipy.ndimage import binary_erosion  # Import binary_erosion for morphological processing

# Input and output parent folders
input_parent = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\7-PhaseExtractedTomograms\Sw2"  # Input directory containing TIFF stacks
output_parent = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\8-EdgeImprovedTomograms\Sw2"  # Output directory for processed TIFF stacks

def process_and_save_tomograms(input_parent, output_parent):  # Define function to process and save tomograms
    for root, dirs, files in os.walk(input_parent):  # Recursively traverse the input directory
        # Collect all .tif files in the current directory
        tif_files = [os.path.join(root, f) for f in files if f.endswith(".tif")]  # List full paths of TIFF files
        if not tif_files:  # Check if there are no TIFF files
            continue  # Skip to the next directory

        # Sort the tif_files to maintain order
        tif_files.sort()  # Sort files to ensure consistent slice ordering

        # Determine the subfolder name relative to the input parent
        relative_path = os.path.relpath(root, input_parent)  # Get path relative to input_parent
        parent_folder, subfolder = os.path.split(relative_path)  # Split into parent and subfolder names
        if not subfolder:  # Check if there’s no valid subfolder (e.g., root is input_parent)
            continue  # Skip to the next directory

        # Print when processing a subfolder
        print(f"Processing {parent_folder}/{subfolder}...")  # Log the subfolder being processed

        # Load all slices into a 3D stack
        slices = [imageio.imread(f) for f in tif_files]  # Read each TIFF file as a 2D slice
        stack = np.stack(slices, axis=0)  # Stack slices into a 3D array (depth, height, width)

        # Erode the middle part of the stack with full padding
        padded_stack = np.pad(stack, pad_width=1, mode='constant', constant_values=0)  # Add 1-voxel padding with zeros
        binary_mask_padded = binary_erosion(padded_stack > 0, structure=np.ones((3, 3, 3)))  # Apply 3D erosion to non-zero regions
        binary_mask = binary_mask_padded[1:-1, 1:-1, 1:-1]  # Remove padding to match original dimensions

        # Erode the first and last slices with a smaller structuring element
        binary_mask[0] = binary_erosion(stack[0] > 0, structure=np.ones((3, 3)))  # Apply 2D erosion to the first slice
        binary_mask[-1] = binary_erosion(stack[-1] > 0, structure=np.ones((3, 3)))  # Apply 2D erosion to the last slice
        
        # Apply the mask to the original stack to extract voxel values
        extracted_values_stack = stack * binary_mask.astype(stack.dtype)  # Multiply stack by mask to keep valid voxels

        # Ensure the output folder exists
        output_folder = os.path.join(output_parent, relative_path)  # Construct output folder path
        os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn’t exist

        # Save each extracted value slice back as a 2D .tif file
        for i, slice in enumerate(extracted_values_stack):  # Iterate over slices
            output_file = os.path.join(output_folder, f"slice{i:04d}.tif")  # Create output filename with zero-padded index
            imageio.imwrite(output_file, slice)  # Save the slice as a TIFF file

        # Print when the subfolder is finished
        print(f"Finished {parent_folder}/{subfolder}.")  # Log completion of the subfolder

    # Print when all folders have been processed
    print("✅ Processing complete.")  # Log completion of all processing

if __name__ == "__main__":  # Check if the script is run directly
    process_and_save_tomograms(input_parent, output_parent)  # Call the processing function with input and output paths