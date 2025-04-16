"""
Code Description:
This script applies binary masks to tomogram TIFF images to extract regions of interest, preserving the original data type. It:
1. Recursively traverses a base directory to process tomogram and mask TIFF files in corresponding subfolders.
2. Applies masks (valued at 255) to tomograms via element-wise multiplication to isolate specific regions.
3. Saves the masked tomograms in a mirrored output directory structure, maintaining 16-bit data type.
Key features:
- Ensures mask and tomogram files are aligned by filename for accurate processing.
- Handles missing mask files gracefully by skipping unmatched tomograms.
- Supports folder hierarchies with automatic output directory creation.
Prerequisites:
- Input TIFF files must be readable by imageio and represent single-band image data.
Dependencies: os, numpy, imageio (install via pip install numpy imageio).

Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, Dec 2024.
"""

import os  # Import os module for file and directory operations
import numpy as np  # Import NumPy for numerical operations and array handling
import imageio.v2 as imageio  # Import imageio.v2 for reading and writing TIFF files

# Define base directories for tomograms, masks, and output masked tomograms
tomogram_base_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\1-CroppedTomograms\Sw2"  # Directory for input tomogram TIFFs
mask_base_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\2-MaskTomograms\1-ExteriorMask\Sw2"  # Directory for mask TIFFs
maskedtomogram_base_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\3-MaskedTomograms\Sw2"  # Output directory for masked tomograms

# Ensure the output directory exists
os.makedirs(maskedtomogram_base_dir, exist_ok=True)  # Create output base directory if it doesn’t exist

print("Starting tomogram processing:")  # Log the start of processing
# Loop through each folder in the tomogram base directory
for folder_name in os.listdir(tomogram_base_dir):  # Iterate over items in the tomogram base directory
    print(f"Processing {folder_name}...")  # Log the current folder being processed
    tomogram_folder = os.path.join(tomogram_base_dir, folder_name)  # Construct path to current tomogram subfolder
    mask_folder = os.path.join(mask_base_dir, folder_name)  # Construct path to corresponding mask subfolder
    masked_tomogram_folder = os.path.join(maskedtomogram_base_dir, folder_name)  # Construct path for output masked tomograms

    # Ensure the current folder is a directory and exists in both tomograms and masks
    if os.path.isdir(tomogram_folder) and os.path.isdir(mask_folder):  # Verify both folders exist and are directories
        os.makedirs(masked_tomogram_folder, exist_ok=True)  # Create output subfolder if it doesn’t exist

        # Loop through each tomogram file in the folder
        for file_name in os.listdir(tomogram_folder):  # Iterate over files in the tomogram subfolder
            tomogram_path = os.path.join(tomogram_folder, file_name)  # Construct full path to tomogram file
            mask_path = os.path.join(mask_folder, file_name)  # Construct full path to corresponding mask file
            masked_tomogram_path = os.path.join(masked_tomogram_folder, file_name)  # Construct path for output masked tomogram

            # Ensure the mask file exists before processing
            if os.path.exists(mask_path):  # Check if the mask file exists
                # Read the tomogram and mask files
                tomogram = imageio.imread(tomogram_path).astype(np.uint16)  # Load tomogram and convert to 16-bit unsigned integer
                mask = imageio.imread(mask_path).astype(np.uint8)  # Load mask and convert to 8-bit unsigned integer

                # Convert mask to binary (1 for 255 and 0 for other values), ensuring compatibility with 16-bit tomogram
                mask = (mask == 255).astype(np.uint16)  # Create binary mask (1 where mask is 255, 0 elsewhere) as 16-bit

                # Apply the mask to the tomogram by element-wise multiplication
                masked_tomogram = tomogram * mask  # Multiply tomogram by mask to retain only masked regions

                # Save the masked tomogram as a 16-bit image
                imageio.imwrite(masked_tomogram_path, masked_tomogram)  # Save masked tomogram to output path
                
    print(f"Processing complete and masked tomogram saved for {folder_name}.")  # Log completion for the folder
    
# Print completion message to indicate successful processing
print("✅ Processing complete.")  # Log overall completion of processing