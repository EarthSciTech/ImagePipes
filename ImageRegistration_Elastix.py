"""
Python script for performing robust 3D image registration using direct Elastix calls.
 Code Description:
 This script performs robust 3D image registration using the Elastix library to align a moving (distorted or misaligned) 
 image stack to a fixed (reference) image stack. It corrects for translations, rotations, and scaling differences.
 The script:
 1. Converts input TIFF stacks to .mhd format for Elastix compatibility.
 2. Executes Elastix with an affine transformation and multi-resolution registration using predefined parameters.
 3. Converts the aligned .mhd result back to TIFF format, preserving original filenames and data types.
 4. Cleans up temporary intermediate files (.mhd and parameter files) after processing.
 Key features:
 - Uses SimpleITK for image format conversions and NumPy for array handling.
 - Employs a robust registration pipeline with the AdvancedMattesMutualInformation metric and adaptive stochastic gradient descent.
 - Maintains original image metadata (e.g., data type) in the output.
 Prerequisites:
 - Elastix must be installed and its executable path specified.
   Elastix is available at: https://elastix.lumc.nl/download.php
 - Input TIFF stacks must be single-channel images in a folder.
 Dependencies: os, subprocess, numpy, tifffile, SimpleITK (install via pip install SimpleITK).
 
 Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, March 2025.
"""

import os            # Import os module for file and directory operations
import subprocess    # Import subprocess module to run external commands (Elastix)
import numpy as np   # Import numpy for numerical array operations
import tifffile      # Import tifffile for reading and writing TIFF files
import SimpleITK as sitk  # Import SimpleITK for image processing and format conversion
import shutil             # Import module for file operations like moving files
import glob               # Import module for matching filenames with patterns

# Paths to image data
path_fixed_image = r'Y:\users\ream\1-RawData\5-AdvectionImages\Series1\Sw1'          # Define path to the fixed (reference) image stack
path_moving_image = r'Y:\users\ream\1-RawData\5-AdvectionImages\Series1\Sw2_2'        # Define path to the moving (to-be-aligned) image stack
output_path_aligned_image = r'Y:\users\ream\1-RawData\5-AdvectionImages\Series1\Sw2_3_aligned'  # Define output path for aligned images

# Set Elastix executable path
elastix_exe = r'Z:\users\ream\0-Software\elastix-5.2.0-win64\elastix.exe'  # Specify the path to the Elastix executable

print('Performing Elastix image registration...')  # Print a message to indicate the start of the registration process

# Create output directory
os.makedirs(output_path_aligned_image, exist_ok=True)  # Create the output directory if it doesn’t already exist

# Function to convert TIFF stack to .mhd
def convert_tiff_to_mhd(tiff_folder, output_filename):  # Define a function to convert TIFF stack to .mhd format
    file_list = sorted([os.path.join(tiff_folder, f) for f in os.listdir(tiff_folder) if f.endswith('.tif')])  # Create a sorted list of TIFF files in the folder
    images = np.stack([tifffile.imread(f) for f in file_list])  # Read each TIFF file and stack them into a 3D NumPy array
    sitk_img = sitk.GetImageFromArray(images)  # Convert the NumPy array to a SimpleITK image object
    sitk.WriteImage(sitk_img, output_filename)  # Write the SimpleITK image to an .mhd file

# Convert TIFF stacks to .mhd format
fixed_mhd = os.path.join(output_path_aligned_image, 'fixed.mhd')  # Define the path for the fixed image in .mhd format
moving_mhd = os.path.join(output_path_aligned_image, 'moving.mhd')  # Define the path for the moving image in .mhd format
convert_tiff_to_mhd(path_fixed_image, fixed_mhd)  # Convert the fixed TIFF stack to .mhd
convert_tiff_to_mhd(path_moving_image, moving_mhd)  # Convert the moving TIFF stack to .mhd

# Define a multi-line string containing Elastix registration parameters
parameter_text = """  
(FixedInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingInternalImagePixelType "float")
(MovingImageDimension 3)
(Registration "MultiResolutionRegistration")
(Interpolator "LinearInterpolator")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(FixedImagePyramid "FixedRecursiveImagePyramid")
(MovingImagePyramid "MovingRecursiveImagePyramid")
(Optimizer "AdaptiveStochasticGradientDescent")
(Transform "EulerTransform")
(Metric "AdvancedMattesMutualInformation")
(AutomaticScalesEstimation "true")
(AutomaticTransformInitialization "true")
(HowToCombineTransforms "Compose")
(NumberOfHistogramBins 64)
(ErodeMask "false")
(NumberOfResolutions 4)
(MaximumNumberOfIterations 1000)
(NumberOfSpatialSamples 20000)
(ImageSampler "RandomCoordinate")
(NewSamplesEveryIteration "true")
(BSplineInterpolationOrder 1)
(FinalBSplineInterpolationOrder 3)
(DefaultPixelValue -1000)
(WriteResultImage "true")
(ResultImagePixelType "float")
(ResultImageFormat "mhd")
"""

# Save parameters to a temporary file
param_file = os.path.join(output_path_aligned_image, 'elastix_parameters.txt')  # Define path for the temporary parameter file
with open(param_file, 'w') as f:  # Open the parameter file in write mode
    f.write(parameter_text)  # Write the parameter text to the file

# Run Elastix
cmd = [  # Create a list of command-line arguments for running Elastix
    elastix_exe,  # Path to the Elastix executable
    '-f', fixed_mhd,  # Specify the fixed image file
    '-m', moving_mhd,  # Specify the moving image file
    '-out', output_path_aligned_image,  # Specify the output directory
    '-p', param_file  # Specify the parameter file
]
subprocess.run(cmd, check=True)  # Execute the Elastix command and raise an error if it fails

# Function to convert .mhd result back to TIFF stack with original names and types preserved
def convert_mhd_to_tiff_preserve_names(mhd_file, original_folder, output_folder):  # Define a function to convert .mhd back to TIFF
    original_files = sorted([f for f in os.listdir(original_folder) if f.endswith('.tif')])  # Get sorted list of original TIFF filenames
    aligned_img = sitk.ReadImage(mhd_file)  # Read the aligned .mhd image into a SimpleITK object
    aligned_stack = sitk.GetArrayFromImage(aligned_img)  # Convert the SimpleITK image to a NumPy array
    original_dtype = tifffile.imread(os.path.join(original_folder, original_files[0])).dtype  # Get the data type of the original TIFF files

    for slice_img, original_name in zip(aligned_stack, original_files):  # Iterate over slices and original filenames
        tifffile.imwrite(  # Write each slice as a TIFF file
            os.path.join(output_folder, original_name),  # Use the original filename in the output folder
            slice_img.astype(original_dtype)  # Convert slice to the original data type
        )

# Convert aligned result back to TIFF stack preserving original filenames and data type
aligned_mhd = os.path.join(output_path_aligned_image, 'result.0.mhd')  # Define path to the aligned .mhd result from Elastix
convert_mhd_to_tiff_preserve_names(aligned_mhd, path_moving_image, output_path_aligned_image)  # Convert aligned .mhd back to TIFF

# Clean up temporary files
os.remove(fixed_mhd)  # Delete the temporary fixed .mhd file
os.remove(moving_mhd)  # Delete the temporary moving .mhd file
os.remove(param_file)  # Delete the temporary parameter file
os.remove(aligned_mhd)  # Delete the aligned result .mhd file
os.remove(fixed_mhd.replace('.mhd', '.raw'))  # Delete the associated fixed .raw data file
os.remove(moving_mhd.replace('.mhd', '.raw'))  # Delete the associated moving .raw data file
os.remove(aligned_mhd.replace('.mhd', '.raw'))  # Delete the associated aligned .raw data file

# Define the full path to a subdirectory named 'log_files' within the output directory
log_files_folder = os.path.join(output_path_aligned_image, 'log_files') 

# Create the log files directory if it doesn't already exist
os.makedirs(log_files_folder, exist_ok=True)

# Move elastix log and iteration files to the log_files folder
# Iterate through patterns of files (*.txt and *.log) to move to the log_files folder
for file_pattern in ['*.txt', '*.log']:
    # Loop over all matching files within the output directory
    for file in glob.glob(os.path.join(output_path_aligned_image, file_pattern)):
        shutil.move(file, log_files_folder) # Move each matched file into the 'log_files' subdirectory

print('✅ Processing complete.')  # Print a message to indicate successful completion