"""
Code Description:
 This script processes TIFF (.tif) files containing raster data (e.g., tomograms) to convert pixel values 
 into concentration values using a linear transformation (y = (x - intercept) / slope), followed by 
 thresholding to zero out low concentrations. It:
 1. Recursively traverses an input folder to process all .tif files in subdirectories.
 2. Applies the transformation only to non-zero pixels, then sets values below a concentration threshold to zero.
 3. Saves the resulting data as new .tif files in a mirrored output folder structure.
 Key features:
 - Maintains original file metadata (except for data type, updated to 32-bit float).
 - Creates output subfolders as needed to replicate the input folder hierarchy.
 - Provides progress feedback via print statements.
 Prerequisites:
 - Input .tif files must be readable by rasterio and contain single-band raster data.
 Dependencies: os, rasterio, numpy (install via pip install rasterio numpy).
 
Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, Feb 2025.
"""

import os  # Module for file and directory operations
import rasterio  # Library for reading and writing raster data
import numpy as np  # Library for numerical operations

# Input folder containing the original TIF files
input_folder = r"Z:\users\ream\2-AnalysedData\3-InertiaEffectData\8-LiquidPhaseEdgeImprovedTomograms_basis"  # Define the input folder path containing TIFF files

# Output folder where processed TIF files will be saved
output_folder = r"Z:\users\ream\2-AnalysedData\3-InertiaEffectData\10-LiquidPhaseConcentration_basis"  # Define the output folder path for processed files

intercept = 13924  # Set the intercept value for the linear transformation
slope = 11390  # Set the slope value for the linear transformation
concentration_threshold = 0.2  # Define an arbitrary concentration threshold below which values are set to zero

# Walk through all directories and files in the input folder
for root, _, files in os.walk(input_folder):  # Use os.walk to recursively traverse the input folder and its subdirectories
    if not files:  # Check if the current directory has no files
        continue  # Skip empty folders to avoid unnecessary processing

    # Extract relative subfolder path
    relative_path = os.path.relpath(root, input_folder)  # Compute the relative path of the current folder from the input folder
    output_subfolder = os.path.join(output_folder, relative_path)  # Construct the corresponding output subfolder path

    # Print message indicating the start of processing a subfolder
    print(f"Processing: {relative_path}...")  # Display a message to indicate processing has started for this subfolder

    # Create the output directory if it doesn't exist
    os.makedirs(output_subfolder, exist_ok=True)  # Create the output subfolder if it doesn’t already exist, no error if it does

    for file in files:  # Iterate over all files in the current subdirectory
        if file.endswith('.tif'):  # Check if the file is a TIFF file
            # Construct full input and output file paths
            input_file_path = os.path.join(root, file)  # Create the full path to the input TIFF file
            output_file_path = os.path.join(output_subfolder, file)  # Create the full path to the output TIFF file

            # Open the TIF file for reading
            with rasterio.open(input_file_path) as src:  # Open the input TIFF file using rasterio in read mode
                # Read the first band (layer) of the raster data as float for calculations
                data = src.read(1).astype(np.float32)  # Read the first band of the TIFF file into a NumPy array as 32-bit float
                profile = src.profile  # Copy the metadata (e.g., dimensions, CRS) from the source file

            # Create an empty array to store concentration values (default is 0)
            concentration_data = np.zeros_like(data, dtype=np.float32)  # Initialize an array of zeros with the same shape and type as the input data

            # Create a mask to identify non-zero pixels
            nonzero_mask = data != 0  # Create a boolean mask where True indicates non-zero pixel values

            # Apply conversion only on non-zero pixels
            concentration_data[nonzero_mask] = (data[nonzero_mask] - intercept) / slope  # Apply linear transformation to non-zero pixels

            # Apply threshold: set concentrations below the threshold to zero
            concentration_data[concentration_data < concentration_threshold] = 0  # Set concentration values below the threshold to zero

            # Update the profile for saving as 32-bit floating point
            profile.update(dtype=rasterio.float32, count=1)  # Update the metadata to specify 32-bit float data type and single band

            # Write the modified data to the new output path
            with rasterio.open(output_file_path, 'w', **profile) as dst:  # Open a new TIFF file for writing with updated metadata
                dst.write(concentration_data, 1)  # Write the processed concentration data to the first band of the output file

    # Print message indicating the subfolder is completely processed
    print(f"Finished processing: {relative_path}\n")  # Display a message to indicate processing is complete for this subfolder

print("✅ Processing complete.")  # Print a final message when all files and folders have been processed