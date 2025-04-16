"""
 Code Description:
 This script applies Non-Local Means (NLM) denoising to TIFF image stacks, either in 2D (slice-by-slice) or 3D (entire volume).
 It:
 1. Recursively traverses an input directory to process TIFF files in subfolders.
 2. Reads images in their native data type, converts to float32 for denoising if not already float, and rescales to the original type.
 3. Saves denoised images in a mirrored output folder structure, preserving the original data type.
 Key features:
 - Supports both 2D and 3D denoising, with user input to choose the mode ('yes' for 3D, 'no' for 2D).
 - Skips empty slices in 2D mode to optimize processing and handles zero-valued images gracefully.
 - Uses parameters aligned with Avizo’s NLM settings for consistency.
 Prerequisites:
 - Input TIFF files must be readable by tifffile and represent single-band image data.
 Dependencies: os, numpy, tifffile, scikit-image (install via pip install tifffile scikit-image).

Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, Feb 2025.
"""

import os  # Import os module for file and directory operations
import numpy as np  # Import numpy for numerical operations
import tifffile as tiff  # Import tifffile for reading and writing TIFF files, aliased as tiff
from skimage.restoration import denoise_nl_means  # Import Non-Local Means denoising function from scikit-image

# Define the input directory where the TIFF files are stored
input_path = r"Z:\\users\\ream\\2-AnalysedData\\3-InertiaEffectData\\8-LiquidPhaseEdgeImprovedTomograms_basis"  # Specify the input directory path

# Define the output directory where results will be saved
output_path = r"Z:\\users\\ream\\2-AnalysedData\\3-InertiaEffectData\\9-LiquidPhaseDenoised_basis"  # Specify the output directory path

# Define parameters for Non-Local Means (NLM) filtering
nlm_params = {  # Create a dictionary of NLM parameters aligned with Avizo’s settings
    "patch_size": 5,  # Set the size of the local neighborhood (e.g., 5x5 pixels for 2D, 5x5x5 for 3D)
    "patch_distance": 10,  # Set the size of the search window (e.g., 10x10 pixels for 2D, 10x10x10 for 3D)
    "search_window_shape": "square",  # Specify the shape of the search window (square for 2D, cube for 3D; disk/ball unavailable)
    "h": 0.05,  # Set the similarity factor controlling denoising strength
    "fast_mode": True  # Enable faster computation for NLM
}

def denoise_image(image, params, is_3d=False, original_dtype=np.uint16):  # Define a function to denoise an image, with original data type
    # Ensure the image is not empty before processing
    if np.count_nonzero(image) == 0:  # Check if the image contains only zeros
        return image  # Return the image unchanged if it’s all zeros
    
    # Convert to float32 if not already a floating-point type
    if not np.issubdtype(image.dtype, np.floating):  # Check if the image is not float (e.g., uint8, uint16)
        image_float = image.astype(np.float32)  # Convert to float32 for processing
    else:  # If already float (e.g., float32, float64)
        image_float = image  # Use the image as is

    # Normalize the image to [0,1] for better denoising
    image_max = np.max(image_float) if np.max(image_float) > 0 else 1  # Get the maximum value, default to 1 to avoid division by zero
    normalized_image = image_float / image_max  # Normalize to [0,1]

    # Apply Non-Local Means denoising
    if is_3d:  # Check if processing in 3D mode
        denoised = denoise_nl_means(normalized_image,  # Apply NLM to the 3D volume
                                    patch_size=params["patch_size"],  # Use specified patch size
                                    patch_distance=params["patch_distance"],  # Use specified patch distance
                                    h=params["h"],  # Use specified similarity factor
                                    fast_mode=params["fast_mode"])  # Use specified fast mode
    else:  # Process in 2D mode (slice-by-slice)
        denoised = np.zeros_like(normalized_image)  # Initialize an output array with the same shape as input
        
        for i in range(image.shape[0]):  # Iterate over slices (first dimension)
            non_zero_slice = image[i] > 0  # Identify non-zero pixels in the current slice
            if np.any(non_zero_slice):  # Check if the slice has any non-zero pixels
                temp_slice = normalized_image[i]  # Extract the current normalized slice
                
                # Apply denoising to the full slice
                denoised_slice = denoise_nl_means(temp_slice,  # Apply NLM to the 2D slice
                                                  patch_size=params["patch_size"],  # Use specified patch size
                                                  patch_distance=params["patch_distance"],  # Use specified patch distance
                                                  h=params["h"],  # Use specified similarity factor
                                                  fast_mode=params["fast_mode"])  # Use specified fast mode
                
                denoised[i] = denoised_slice  # Store the denoised slice
            else:  # If the slice is all zeros
                denoised[i] = normalized_image[i]  # Retain the original (normalized) slice unchanged
    
    # Rescale back to the original range and convert to original data type
    denoised = (denoised * image_max)  # Rescale to the original range
    if np.issubdtype(original_dtype, np.floating):  # If original type was float (e.g., float32, float64)
        denoised = denoised.astype(original_dtype)  # Convert to original float type
    else:  # If original type was integer (e.g., uint8, uint16)
        # Clip to valid range for integer types (e.g., [0, 255] for uint8, [0, 65535] for uint16)
        max_val = np.iinfo(original_dtype).max if np.issubdtype(original_dtype, np.integer) else 1.0
        denoised = np.clip(denoised, 0, max_val).astype(original_dtype)  # Clip and convert to original type
    
    return denoised  # Return the denoised image

# Ask user whether to process the images in 2D or 3D
process_in_3d = input("Process in 3D (yes/no)? ").strip().lower() == 'yes'  # Prompt user for 3D mode; 'yes' sets True (3D), 'no' sets False (2D)

# Walk through the directory structure to process all target folders and TIFF files
for root, dirs, files in os.walk(input_path):  # Use os.walk to recursively traverse the input directory
    # Find all TIFF files in the current folder
    tiff_files = [f for f in files if f.lower().endswith('.tif')]  # Create a list of TIFF files (case-insensitive)

    # Check if the current folder contains TIFF files
    if tiff_files:  # Proceed only if TIFF files are found
        # Identify the current target folder name
        target_folder = os.path.basename(root)  # Extract the name of the current folder

        # Print a message to indicate which folder is being processed
        print(f"Processing folder: {target_folder}...")  # Display a message for the current folder

        # Create an output folder using only the target folder name
        denoised_folder = os.path.join(output_path, target_folder)  # Construct the output folder path
        os.makedirs(denoised_folder, exist_ok=True)  # Create the output folder if it doesn’t exist, no error if it does

        # Initialize a list to store slices for 3D processing
        volume_slices = []  # Create an empty list to collect slices for 3D mode
        dtypes = []  # Initialize a list to store original data types for each slice

        # Process each TIFF file in the current folder
        for tiff_file in sorted(tiff_files):  # Iterate over sorted TIFF files to maintain order
            # Construct the full path to the current TIFF file
            tomogram_path = os.path.join(root, tiff_file)  # Create the full path to the input TIFF file

            # Read the TIFF file and store it as a NumPy array
            image = tiff.imread(tomogram_path)  # Read the TIFF file into a NumPy array (native data type)
            original_dtype = image.dtype  # Store the original data type (e.g., uint8, uint16, float32)

            if process_in_3d:  # Check if processing in 3D mode
                # For 3D processing, add the slice to the volume
                volume_slices.append(image)  # Append the current slice to the volume list
                dtypes.append(original_dtype)  # Store the original data type
            else:  # Process in 2D mode
                # For 2D processing, denoise each slice individually
                denoised_image = denoise_image(np.expand_dims(image, axis=0), nlm_params.copy(), is_3d=False, original_dtype=original_dtype)[0]  # Denoise the slice
                
                # Save the processed slice to the output directory in the original data type
                output_file_path = os.path.join(denoised_folder, tiff_file)  # Construct the output file path
                tiff.imwrite(output_file_path, denoised_image)  # Save the denoised slice (already in original type)

        if process_in_3d:  # Check if processing in 3D mode
            # For 3D processing, denoise the entire volume
            volume = np.stack(volume_slices)  # Stack all slices into a 3D volume
            # Use the most common data type among slices (or first slice’s type if uniform)
            most_common_dtype = max(set(dtypes), key=dtypes.count) if dtypes else np.uint16
            denoised_volume = denoise_image(volume, nlm_params.copy(), is_3d=True, original_dtype=most_common_dtype)  # Apply NLM to the 3D volume
            
            # Save each slice of the denoised volume in the original data type
            for i, slice_image in enumerate(denoised_volume):  # Iterate over slices in the denoised volume
                output_file_path = os.path.join(denoised_folder, tiff_files[i])  # Construct the output file path
                tiff.imwrite(output_file_path, slice_image)  # Save the slice (already in original type)

        # Print a message to confirm the folder's results are saved
        print(f"Folder {target_folder} processed.\n")  # Display a confirmation message

# Print a message to indicate that the entire processing is complete
print("✅ Denoising complete.")  # Print a final message when all processing is done