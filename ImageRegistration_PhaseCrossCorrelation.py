"""
Code Description:
This script performs 3D image registration using phase cross-correlation to align a moving image stack to a fixed image stack. It:
1. Loads TIFF stacks from specified directories for fixed and moving images.
2. Computes the translation vector between stacks using phase cross-correlation and applies it to align the moving stack.
3. Saves the aligned stack in a new directory, preserving the original data type with zlib compression.
Key features:
- Supports any image data type (e.g., uint8, uint16, float32) by preserving the input type.
- Uses high-precision upsampling for accurate shift estimation.
- Processes stacks recursively and maintains slice order in output.
Prerequisites:
- Input TIFF files must be readable by tifffile and represent single-band image data.
Dependencies: os, numpy, tifffile, scikit-image, scipy (install via pip install numpy tifffile scikit-image scipy).

Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, April 2025.
"""

import os  # Import os module for file and directory operations
import numpy as np  # Import NumPy for numerical operations and array handling
import tifffile  # Import tifffile for reading and writing TIFF files
from skimage.registration import phase_cross_correlation  # Import phase_cross_correlation for image registration
from scipy.ndimage import shift  # Import shift for applying translation to images

# Paths
path_fixed = r'Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\2-MaskTomograms\2-SolidMask\Sw1'  # Directory for fixed image stack
path_moving = r'Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\3-MaskedTomograms\Sw2'  # Directory for moving image stack
output_path_aligned = r'Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\2-MaskTomograms\2-SolidMask\AlignedMask\Sw2'  # Output directory for aligned stack

# Ensure output directory exists
os.makedirs(output_path_aligned, exist_ok=True)  # Create output directory if it doesn’t exist

# Load image stacks as 3D numpy arrays
def load_image_stack(path):  # Load all TIFF files in a directory into a 3D array
    file_list = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.tif')])  # Get sorted list of TIFF files
    return np.stack([tifffile.imread(f) for f in file_list])  # Stack images into a 3D array

print('Processing...')  # Log the start of processing
fixed_stack = load_image_stack(path_fixed)  # Load the fixed image stack (reference)
moving_stack = load_image_stack(path_moving)  # Load the moving image stack (to be aligned)

# Compute the translation between the fixed and moving stacks
shift_vector, error, diffphase = phase_cross_correlation(moving_stack, fixed_stack, upsample_factor=10)  # Calculate translation vector
print(f'Shift vector (z, y, x): {shift_vector}')  # Log the computed shift vector

# Apply shift to the moving stack to align with the fixed stack
def align_stack(stack, shift_vector):  # Align a stack by applying a translation
    return shift(stack, shift=shift_vector, mode='nearest', order=0)  # Shift stack using nearest-neighbor interpolation

aligned_stack = align_stack(fixed_stack, shift_vector)  # Align the fixed stack to match the moving stack’s position

# Save aligned stack as a stack of compressed TIFF files
for i, slice_img in enumerate(aligned_stack):  # Iterate over slices in the aligned stack
    tifffile.imwrite(
        os.path.join(output_path_aligned, f'slice{i:04d}.tif'),  # Construct output file path with zero-padded index
        slice_img.astype(fixed_stack.dtype),  # Preserve the original data type of the fixed stack
        compression='zlib'  # Use zlib for efficient lossless compression
    )

print('✅ Processing complete.')  # Log completion of processing