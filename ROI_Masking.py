"""
Code Description:
This script generates 8-bit binary masks for 3D TIFF stacks using a 2D Region of Interest (ROI) defined in an ImageJ .roi file. It:
1. Loads the ROI to create a binary mask matching the dimensions of input TIFF slices.
2. Processes TIFF stacks in a root folder or its subfolders, generating a mask for each slice.
3. Saves masks as 8-bit TIFF files (255 inside ROI, 0 outside) in a mirrored output directory structure.
Key features:
- Supports single-stack (TIFFs in root) or multi-stack (TIFFs in subfolders) processing.
- Uses zlib compression for efficient storage of output masks.
- Maintains folder hierarchy and ensures consistent slice ordering.
Prerequisites:
- Input TIFF files must be readable by tifffile; ROI file must be in ImageJ .roi format.
Dependencies: os, numpy, tifffile, roifile, scikit-image (install via pip install numpy tifffile roifile scikit-image).

Code by: Amirsaman Rezaeyan, amirsaman[dot]rezaeyan[at sign]gmail[dot]com, Zürich, Switzerland, April 2025.
"""

import os  # Import os module for file and directory operations
import numpy as np  # Import NumPy for numerical operations and array handling
import tifffile  # Import tifffile for reading and writing TIFF files
from roifile import ImagejRoi  # Import ImagejRoi for parsing ImageJ ROI files
from skimage.draw import polygon2mask  # Import polygon2mask to create binary masks from polygon vertices

# === User parameters ===
ROI_PATH = r"Z:\users\ream\2-AnalysedData\3-InertiaEffectData\0-ROI\RadialMask.roi"  # Path to ImageJ ROI file defining mask region
INPUT_ROOT = r"Z:\users\ream\2-AnalysedData\3-InertiaEffectData\Testcropping"  # Root folder containing input TIFF stacks
OUTPUT_ROOT = r"Z:\users\ream\2-AnalysedData\3-InertiaEffectData\testmask"  # Root folder for output mask TIFFs
# =======================

def load_roi_mask(roi_path, image_shape):  # Load ImageJ ROI and generate 8-bit mask
    """
    Load ImageJ .roi, compute a binary mask of full image_shape.
    Returns an 8-bit uint8 mask array (0 outside, 255 inside ROI).
    """
    roi = ImagejRoi.fromfile(roi_path)  # Parse the ROI file
    coords = roi.coordinates()  # Get Nx2 array of (x, y) vertices
    # Extract polygon vertices in (row, col) order
    polygon = np.vstack([coords[:,1], coords[:,0]]).T  # Convert to (row, col) format
    # Build boolean mask for full image
    mask_bool = polygon2mask(image_shape, polygon)  # Create binary mask (True inside ROI)
    # Convert to 8-bit: inside ROI = 255, outside = 0
    mask_uint8 = (mask_bool.astype(np.uint8)) * 255  # Scale boolean mask to 8-bit (0 or 255)
    return mask_uint8  # Return 8-bit mask

def process_folder(input_dir, output_base, roi_path):  # Generate masks for TIFF slices in a folder
    """
    For every TIFF slice in input_dir, generate and save an 8-bit ROI mask.
    Mirrors folder structure under output_base, naming masks slice0000.tif, etc.
    """
    # List and sort TIFF files
    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".tiff"))]  # List TIFF files
    files.sort()  # Sort files to maintain slice order

    # Create corresponding output directory
    rel = os.path.relpath(input_dir, INPUT_ROOT)  # Compute path relative to input root
    out_dir = os.path.join(output_base, rel)  # Derive corresponding output folder
    os.makedirs(out_dir, exist_ok=True)  # Create output folder if it doesn’t exist

    print(f"  → Creating masks for {len(files)} slices in: {input_dir}")  # Log number of slices and input directory
    for idx, fname in enumerate(files):  # Iterate over sorted TIFF files
        img_path = os.path.join(input_dir, fname)  # Construct path to TIFF file
        # Read one slice to get its shape
        img = tifffile.imread(img_path)  # Load TIFF slice
        # Generate full-image mask
        mask = load_roi_mask(roi_path, img.shape)  # Create 8-bit mask for slice dimensions
        # Format output filename
        out_fname = f"slice{idx:04d}.tif"  # Create zero-padded slice filename
        # Save as 8-bit TIFF
        tifffile.imwrite(os.path.join(out_dir, out_fname), mask, dtype=np.uint8, compression='zlib')  # Save mask with zlib compression

    print(f"  ✔ Finished masks for folder: {input_dir}\n")  # Log completion for the folder

if __name__ == '__main__':  # Main execution block
    # Determine which folders to process (root or its subfolders)
    entries = os.listdir(INPUT_ROOT)  # List entries under input root
    # Check if root contains TIFFs
    tifs_root = [f for f in entries if f.lower().endswith((".tif", ".tiff"))]  # Check for TIFFs in root
    if tifs_root:  # If TIFFs are found in root
        folders = [INPUT_ROOT]  # Process root as a single stack
    else:
        folders = []  # Initialize empty list for subfolders
        for name in entries:  # Iterate over root entries
            full = os.path.join(INPUT_ROOT, name)  # Construct full path
            if os.path.isdir(full):  # Check if entry is a directory
                if any(f.lower().endswith((".tif", ".tiff")) for f in os.listdir(full)):  # Check for TIFFs
                    folders.append(full)  # Add subfolder to process

    print(f"Starting mask generation for {len(folders)} folder(s)...\n")  # Log number of folders to process
    for fld in folders:  # Iterate over folders to process
        print(f"Processing folder: {fld}")  # Log current folder
        process_folder(fld, OUTPUT_ROOT, ROI_PATH)  # Generate masks for folder

    print("✅ All done! Masks generated.")  # Log overall completion