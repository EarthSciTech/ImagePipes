"""
Code Description:
This script crops 3D TIFF stacks using a 2D Region of Interest (ROI) defined in an ImageJ .roi file. It:
1. Loads the ROI to create a binary mask and compute its bounding box offsets.
2. Processes TIFF stacks in a root folder or its subfolders, cropping each slice to the ROI’s bounding box and masking non-ROI areas.
3. Saves cropped slices as sequentially named TIFF files in a mirrored output directory structure.
Key features:
- Supports both single-stack (TIFFs in root) and multi-stack (TIFFs in subfolders) processing.
- Preserves original pixel data within the ROI, setting non-ROI pixels to zero.
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
ROI_PATH = r"Z:\users\ream\2-AnalysedData\3-InertiaEffectData\0-ROI\SquareCut_Re.roi"  # Path to ImageJ ROI file defining crop region
INPUT_ROOT = r"Z:\users\ream\1-RawData\Croptest"  # Root folder containing input TIFF stacks
OUTPUT_ROOT = r"Z:\users\ream\2-AnalysedData\3-InertiaEffectData\Testcropping"  # Root folder for cropped output TIFFs
# =======================

def load_roi(roi_path):  # Load ImageJ ROI and compute mask and offsets
    """
    Load an ImageJ .roi file, compute the polygon mask and its bounding box offsets.
    Returns:
        mask: 2D boolean numpy array of ROI region (True inside ROI)
        x0, y0: top-left coordinate offsets to apply when cropping each slice
    """
    roi = ImagejRoi.fromfile(roi_path)  # Read the ROI file
    coords = roi.coordinates()  # Get Nx2 array of (x, y) vertices in image coordinates
    # Extract bounding box coordinates
    left, top, right, bottom = roi.left, roi.top, roi.right, roi.bottom  # Get ROI bounds
    # Compute shape of bounding box region
    height = int(bottom - top)  # Calculate number of rows
    width = int(right - left)  # Calculate number of columns
    # Shift polygon vertices to bounding box coordinate system
    polygon = np.vstack([coords[:,1] - top, coords[:,0] - left]).T  # Convert to (row, col) format
    # Build a boolean mask of the polygon area
    mask = polygon2mask((height, width), polygon)  # Create binary mask (True inside ROI)
    return mask, left, top  # Return mask and top-left offsets

def process_stack(input_dir, output_base, mask, x0, y0):  # Crop TIFF stack using ROI mask
    """
    Crop every TIFF slice in input_dir by the ROI mask, then save into output_base,
    naming them slice0000.tif, slice0001.tif, …
    """
    # Find all TIFF files in the directory
    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".tif", ".tiff"))]  # List TIFF files
    files.sort()  # Sort files to maintain slice order

    # Recreate relative folder structure in output
    rel = os.path.relpath(input_dir, INPUT_ROOT)  # Compute path relative to input root
    out_dir = os.path.join(output_base, rel)  # Derive corresponding output folder
    os.makedirs(out_dir, exist_ok=True)  # Create output folder if it doesn’t exist

    print(f"  → Cropping {len(files)} slices in: {input_dir}")  # Log number of slices and input directory
    for idx, fname in enumerate(files):  # Iterate over sorted TIFF files
        img = tifffile.imread(os.path.join(input_dir, fname))  # Read the TIFF slice
        # Crop image to bounding box of ROI
        cropped = img[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]  # Crop to ROI bounding box
        # Zero out pixels outside ROI mask if ROI is non-rectangular
        cropped_masked = np.where(mask, cropped, 0)  # Apply mask, setting non-ROI pixels to 0
        # Format output filename as slice####.tif
        out_fname = f"slice{idx:04d}.tif"  # Create zero-padded slice filename
        # Save the cropped slice to the output directory
        tifffile.imwrite(os.path.join(out_dir, out_fname), cropped_masked)  # Save cropped slice

    print(f"  ✔ Finished folder: {input_dir}\n")  # Log completion for the folder

if __name__ == '__main__':  # Main execution block
    # Load the ROI mask and its offset coordinates
    mask, x0, y0 = load_roi(ROI_PATH)  # Load ROI mask and offsets

    # Determine which folders to process
    entries = os.listdir(INPUT_ROOT)  # List entries under input root
    # Check for TIFF files at root level
    tifs_here = [f for f in entries if f.lower().endswith((".tif", ".tiff"))]  # Check for TIFFs in root

    if tifs_here:  # If TIFFs are found in root
        folders_to_do = [INPUT_ROOT]  # Process root as a single 3D stack
    else:
        folders_to_do = []  # Initialize empty list for subfolders
        for name in entries:  # Iterate over root entries
            full = os.path.join(INPUT_ROOT, name)  # Construct full path
            if os.path.isdir(full):  # Check if entry is a directory
                # If subfolder contains TIFFs, add to list
                if any(f.lower().endswith((".tif", ".tiff")) for f in os.listdir(full)):  # Check for TIFFs
                    folders_to_do.append(full)  # Add subfolder to process

    print(f"Starting cropping for {len(folders_to_do)} stack(s)...\n")  # Log number of stacks to process
    for fld in folders_to_do:  # Iterate over folders to process
        print(f"Processing folder: {fld}")  # Log current folder
        process_stack(fld, OUTPUT_ROOT, mask, x0, y0)  # Process the stack

    print("✅ All done! Cropping complete.")  # Log overall completion