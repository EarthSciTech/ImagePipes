"""
Code Description:
This script processes tomogram slices to segment and extract solid, liquid, and air phases based on cluster values, filling holes and validating interfaces. It:
1. Loads tomogram, mask, and clustered slices from specified directories for multiple folders.
2. Assigns phase labels using cluster values, fills holes using a dominant phase approach, and validates interface voxels against intensity ranges.
3. Saves combined segmented volumes, individual phase masks, and extracted tomogram intensities in a structured output directory.
Key features:
- Supports multiple folders with folder-specific cluster values for solid, liquid, and air phases.
- Uses a neighborhood-based hole-filling method with configurable Manhattan distance.
- Validates phase assignments at interfaces to ensure consistency with tomogram intensities.
Prerequisites:
- Input TIFF files must be readable by OpenCV, single-band, and aligned across tomogram, mask, and cluster directories.
Dependencies: os, opencv-python, numpy, scipy (install via pip install opencv-python numpy scipy).

Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, Dec 2024.
"""

import os  # Import the os module for file and directory operations
import cv2  # Import OpenCV for reading, processing, and writing image files
import numpy as np  # Import NumPy for efficient numerical and array operations
from scipy.ndimage import generic_filter  # Import generic_filter for applying neighborhood-based operations

# Define the base directory paths where data is stored and results will be saved
tomogram_base_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\3-MaskedTomograms"  # Path to raw tomogram slices
mask_base_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\2-MaskTomograms\3-NonsolidMask"  # Path to mask files indicating regions of interest
cluster_base_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\4-NonsolidClusteredTomograms"  # Path to clustered tomogram slices
all_segmented_base_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\5-FullySegmentedTomograms"  # Output directory for combined hole-filled tomograms
phase_segmented_base_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\6-PhaseSegmentedTomograms"  # Output directory for phase-segmented tomograms
extracted_base_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\7-PhaseExtractedTomograms"  # Output directory for extracted tomogram values

# Unified structure: Folder-wise cluster values for each phase
cluster_phase_values = {
    #"PB": {
    #    "solid": [11796, 12736, 13436, 14148], # Solid phase cluster values for "PB"
    #    "liquid": [], # No liquid phase cluster values for "PB"
    #    "air": [8813, 9735, 10729] # Air phase cluster values for "PB"
    #},
    #"KInoPB0_1M": {
    #    "solid": [],
    #    "liquid": [11882.122, 12891.844, 13901.827],
    #    "air": []
    #},
    #"KInoPB1_5M": {
    #    "solid": [],
    #    "liquid": [25440, 33619, 36042],
    #    "air": []
    #},
    #"KInoPB3_0M": {
    #    "solid": [],
    #    "liquid": [52485, 56299],
    #    "air": []
    #},
    #"KIPB0_1M": {
     #   "solid": [16304, 17353, 18211, 19053, 20134],
      #  "liquid": [12839, 14074, 15141],
       # "air": [10270]
    #},
    #"KIPB0_5M": {
     #   "solid": [20589, 20672, 20755, 20836, 20916, 20994, 21071, 21147, 21223, 21299, 21376, 21455, 21537, 21624, 21714, 21809, 21909, 22015, 22126, 22242, 22363, 22489, 22621, 22760, 22906, 23065, 23240, 23440, 23676, 23970, 24359, 24899, 25577, 26307, 27067, 27844, 28624, 29420, 30291, 31413],
      #  "liquid": [16504, 16782, 17035, 17263, 17468, 17653, 17820, 17974, 18117, 18252, 18379, 18498, 18611, 18718, 18819, 18915, 19007, 19094, 19175, 19253, 19328, 19400, 19469, 19536, 19602, 19669, 19737, 19806, 19877, 19949, 20024, 20100, 20179, 20258, 20339, 20420, 20504],
       # "air": [7484, 8309, 8892, 9379, 9818, 10231, 10628, 11016, 11395, 11763, 12117, 12457, 12781, 13096, 13403, 13706, 14009, 14314, 14622, 14936, 15253, 15574, 15893, 16205]
    #},
    #"KIPB1_0M": {
     #   "solid": [15114, 17927, 19579, 20961, 22355],
      #  "liquid": [23889, 25613, 27733],
       # "air": [10694]
    #},
    #"KIPB1_5M": {
     #  "solid": [16943, 19655, 21511, 23359, 25491],
    #    "liquid": [27825, 30267, 32931],
     #   "air": [12309]
   # },
   # "KIPB2_0M": {
    #    "solid": [17925, 20887, 23101, 25693],
     #   "liquid": [28916, 32551, 36438, 40385],
      #  "air": [12268]
    #},
   #"KIPB3_0M": {
    #    "solid": [19623, 22911, 26049, 30031, 34676],
     #   "liquid": [34676, 39574, 44388, 49247],
      #  "air": [13509]
    #},
    #"Re_1_0_0001": {
        # "solid": [19649, 20567, 21349, 22142, 23159, 29884],
       #  "liquid": [12163, 13928, 14989, 15966, 17090, 18420],
     #    "air": []
    # },
    #"Re_1_0_0001": {
    #     "solid": [],
    #     "liquid": [13165, 14172, 14943, 15655, 16391, 17202, 18071, 18983],
    #     "air": [11231]
    # },
    #"Re_2_0_001": {
    #     "solid": [],
    #     "liquid": [13436, 14400, 15229, 16077, 17065, 18275, 19684, 21417],
    #     "air": [11890]
    # },    
    #"Re_3_0_01": {
    #     "solid": [],
    #     "liquid": [13980, 15136, 16370, 17878, 19601, 21582, 24954, 31364],
    #     "air": [12449]
    # },
    #"Re_4_0_05": {
    #     "solid": [],
    #     "liquid": [14065, 15273, 16653, 18386, 20277, 22369, 26595, 33549],
    #     "air": [12525]
    # },
    #Re_5_0_1": {
    #     "solid": [],
    #     "liquid": [14229, 15614, 17190, 19045, 20950, 23113, 27677, 35321],
    #     "air": [12523]
    # },
    #"KIG1_0_1M_KI": {
    #     "solid": [],
    #     "liquid": [13404, 14112, 14744, 15373, 16051, 16830, 17763],
    #     "air": [11120, 12487]
    # },
    #"KIG2_0_35M_KI": {
    #     "solid": [],
    #     "liquid": [16096, 17586, 18671, 19646, 20680, 22039],
    #     "air": [8410, 11721, 13968]
    # },
    #"KIG3_0_7M_KI": {
    #     "solid": [],
    #     "liquid": [19796, 22087, 23564, 24796, 26032, 27576],
    #     "air": [10073, 13683, 16843]
    # },
    #"KIG4_1_05M_KI": {
    #     "solid": [],
    #     "liquid": [23433, 26320, 28345, 30084, 31720, 33612],
    #     "air": [12042, 15695, 19364]
    # },
    #"KIG5_1_4M_KI": {
    #     "solid": [],
    #     "liquid": [28302, 31181, 33514, 35565, 37425, 39545],
    #     "air": [13612, 18469, 23767]
    # },
    #"KIG6_2_0M_KI": {
    #     "solid": [],
    #     "liquid": [32301, 36485, 39964, 43059, 45692, 48412],
    #     "air": [14053, 19446, 26114]
    # },
    #"Sw1": {
    #     "solid": [],
    #     "liquid": [8413, 11449, 13282, 14437, 15433, 16505, 17807, 19517, 22840],
    #     "air": []
    # },
    "Sw2": {
         "solid": [],
         "liquid": [12451, 14225, 15498, 16841, 18488, 20896, 26072],
         "air": [8116, 9903]
     },
    #"Sw3": {
    #     "solid": [],
    #     "liquid": [12639, 14219, 15416, 16651, 18187, 20378, 24305],
    #     "air": [8222, 10185]
    # },
    #"Sw4": {
    #     "solid": [],
    #     "liquid": [11752, 13600, 14785, 15903, 17248, 19018, 21781, 28022],
    #     "air": [8621]
    # },
}

# Define the size of the neighborhood for hole-filling (in voxels)
manhattan = 1  # Number of voxels to search in x, y, z directions (1 voxel each way)
neighbouring_voxels = 2 * manhattan + 1  # Total size of the neighborhood (e.g., 3 for manhattan=1)

# Dynamically extract folder names from cluster_phase_values dictionary keys
folders = list(cluster_phase_values.keys())  # List of folders to process (e.g., ["Sw2"])

# Function to compute the intensity ranges for each phase from cluster values and tomogram intensities
def compute_phase_ranges(tomogram_volume, cluster_phase_values):  # Compute intensity ranges for phases
    print("Computing phase intensity ranges...")  # Log the start of range computation
    phase_ranges = {}  # Initialize dictionary to store phase ranges for each folder
    for folder, phases in cluster_phase_values.items():  # Iterate over folders and their phase configurations
        phase_ranges[folder] = {}  # Initialize sub-dictionary for the current folder
        all_tomogram_values = tomogram_volume.flatten()  # Flatten tomogram volume to get all intensities
        min_intensity = np.min(all_tomogram_values)  # Find the minimum intensity in the tomogram
        max_intensity = np.max(all_tomogram_values)  # Find the maximum intensity in the tomogram

        # Filter out empty phases (those with no cluster values)
        sorted_phases = [p for p in phases if phases[p]]  # Keep phases with non-empty cluster values
        sorted_phases = sorted(sorted_phases, key=lambda p: np.min(phases[p]))  # Sort by minimum cluster value

        # Determine lower and upper limits for each phase
        for i, phase in enumerate(sorted_phases):  # Iterate over sorted phases
            cluster_values = phases[phase]  # Get cluster values for the current phase
            if i == 0:  # For the first phase (lowest intensity)
                lower_limit = min_intensity  # Set lower limit to tomogram’s minimum intensity
            else:  # For subsequent phases
                lower_limit = min(cluster_values)  # Use minimum cluster value as lower limit
                
            if i == len(sorted_phases) - 1:  # For the last phase (highest intensity)
                upper_limit = max_intensity  # Set upper limit to tomogram’s maximum intensity
            else:  # For intermediate phases
                upper_limit = max(cluster_values)  # Use maximum cluster value as upper limit
                               
            phase_ranges[folder][phase] = (lower_limit, upper_limit)  # Store the range for the phase

    return phase_ranges  # Return the computed phase ranges

# Function to determine the dominant phase in a neighborhood window
def dominant_phase(window):  # Compute the most frequent non-zero value in a window
    unique, counts = np.unique(window, return_counts=True)  # Count occurrences of unique values
    if 0 in unique:  # If zero values (holes or background) are present
        counts[unique == 0] = 0  # Ignore zeros in dominant phase calculation
    return unique[np.argmax(counts)]  # Return the value with the highest count (dominant phase)

# Function to fill holes using the dominant phase approach
def correct_holes(volume):  # Fill holes in a segmented volume
    print("Filling holes...")  # Log the start of hole-filling
    filled_volume = volume.copy()  # Create a copy to avoid modifying the input volume

    # Apply generic filter to fill holes across the entire volume
    filled_volume = generic_filter(filled_volume, dominant_phase, size=neighbouring_voxels, mode="constant", cval=0)  # Use dominant phase in neighborhood

    # Handle boundary slices (first and last) with reflection padding to avoid edge artifacts
    first_slice_padded = np.pad(filled_volume[0], pad_width=((1, 1), (1, 1)), mode="reflect")  # Pad first slice (y, x)
    filled_volume[0] = generic_filter(first_slice_padded, dominant_phase, size=neighbouring_voxels, mode="constant", cval=0)[1:-1, 1:-1]  # Update first slice

    last_slice_padded = np.pad(filled_volume[-1], pad_width=((1, 1), (1, 1)), mode="reflect")  # Pad last slice (y, x)
    filled_volume[-1] = generic_filter(last_slice_padded, dominant_phase, size=neighbouring_voxels, mode="constant", cval=0)[1:-1, 1:-1]  # Update last slice

    return filled_volume  # Return the hole-filled volume

# Function to validate only interface voxels based on tomogram intensity ranges
def validate_filled_volume(filled_volume, original_volume, phase_ranges):  # Validate phase assignments at interfaces
    print("Validating filled voxels (holes) at phase interfaces...")  # Log the validation step
    for z in range(filled_volume.shape[0]):  # Iterate over slices
        for y in range(1, filled_volume.shape[1] - 1):  # Iterate over rows, excluding edges
            for x in range(1, filled_volume.shape[2] - 1):  # Iterate over columns, excluding edges
                phase_label = filled_volume[z, y, x]  # Get the phase label of the current voxel
                voxel_intensity = original_volume[z, y, x]  # Get the corresponding tomogram intensity

                if phase_label == 0:  # Skip background (zero) voxels
                    continue

                # Check the 3x3 neighborhood to identify interfaces
                neighborhood = filled_volume[z, y - 1:y + 2, x - 1:x + 2]  # Extract 3x3 patch
                unique_labels = np.unique(neighborhood)  # Get unique phase labels in the patch

                # Validate only if the voxel is at an interface (multiple labels present)
                if len(unique_labels) > 1:  # Interface detected
                    current_phase = sorted_phases[phase_label - 1]  # Map label (1, 2, ...) to phase name
                    lower_limit, upper_limit = phase_ranges[current_phase]  # Get intensity range for the phase

                    # Check if the voxel’s intensity is outside the phase’s range
                    if not (lower_limit <= voxel_intensity <= upper_limit):  # Invalid intensity
                        for phase, (lower_limit, upper_limit) in phase_ranges.items():  # Check other phases
                            if lower_limit <= voxel_intensity <= upper_limit:  # Find matching phase
                                new_label = sorted_phases.index(phase) + 1  # Get corresponding label
                                filled_volume[z, y, x] = new_label  # Update voxel with correct label
                                break

    return filled_volume  # Return the validated volume

# Start the tomogram processing for each folder
print("Starting tomogram processing...")  # Log the start of processing
for folder in folders:  # Iterate over each folder (e.g., Sw2)
    print(f"Processing {folder}...")  # Log the current folder being processed

    # Define input and output paths for the current folder
    cluster_folder = os.path.join(cluster_base_dir, folder)  # Path to clustered tomogram slices
    mask_folder = os.path.join(mask_base_dir, folder)  # Path to mask tomogram slices
    tomogram_folder = os.path.join(tomogram_base_dir, folder)  # Path to raw tomogram slices
    all_segment_folder = os.path.join(all_segmented_base_dir, folder)  # Output path for combined segmented tomograms
    phase_segment_folder = os.path.join(phase_segmented_base_dir, folder)  # Output path for phase-segmented masks
    extracted_folder = os.path.join(extracted_base_dir, folder)  # Output path for extracted intensities

    # Check for missing input directories
    if not os.path.exists(cluster_folder) or not os.path.exists(mask_folder) or not os.path.exists(tomogram_folder):  # Validate input paths
        print(f"Skipping folder {folder}: Missing necessary folder.")  # Log if any directory is missing
        continue  # Skip to the next folder

    # Create output directories if they don’t exist
    os.makedirs(all_segment_folder, exist_ok=True)  # Ensure directory for combined tomograms exists
    os.makedirs(phase_segment_folder, exist_ok=True)  # Ensure directory for phase masks exists
    os.makedirs(extracted_folder, exist_ok=True)  # Ensure directory for extracted intensities exists

    # Load tomogram, mask, and cluster slices
    slice_files = sorted([f for f in os.listdir(cluster_folder) if f.endswith(".tif")])  # Get sorted list of TIFF files
    cluster_volume = np.array([cv2.imread(os.path.join(cluster_folder, f), cv2.IMREAD_UNCHANGED) for f in slice_files])  # Load cluster slices as volume
    mask_volume = np.array([cv2.imread(os.path.join(mask_folder, f), cv2.IMREAD_UNCHANGED) for f in slice_files])  # Load mask slices as volume
    tomogram_volume = np.array([cv2.imread(os.path.join(tomogram_folder, f), cv2.IMREAD_UNCHANGED) for f in slice_files])  # Load raw tomogram slices

    mask_indices = mask_volume == 255  # Create boolean mask for ROI (255 indicates region of interest)

    # Compute phase intensity ranges for the current folder
    phase_ranges = compute_phase_ranges(tomogram_volume, cluster_phase_values)  # Calculate intensity ranges for phases

    # Initialize phase volumes with zeros for each phase
    phase_volumes = {phase: np.zeros_like(cluster_volume) for phase in cluster_phase_values[folder]}  # Create empty volumes for each phase

    # Assign cluster-based labels to phase volumes
    for phase, values in cluster_phase_values[folder].items():  # Iterate over phases and cluster values
        for value in values:  # For each cluster value
            phase_volumes[phase][np.logical_and(mask_indices, cluster_volume == value)] = 255  # Mark matching voxels as 255

    # Combine phases and assign numeric labels
    print("Combining and labeling phases...")  # Log phase combination step
    combined_volume = np.zeros_like(cluster_volume, dtype=np.uint8)  # Initialize combined volume as uint8
    phase_means = {phase: np.mean(tomogram_volume[np.logical_and(mask_indices, phase_volumes[phase] == 255)]) for phase in phase_volumes if np.any(phase_volumes[phase])}  # Compute mean intensity per phase
    sorted_phases = sorted(phase_means, key=phase_means.get)  # Sort phases by mean intensity (low to high)
    for i, phase in enumerate(sorted_phases):  # Assign numeric labels (1, 2, ...)
        combined_volume[phase_volumes[phase] == 255] = i + 1  # Label phase voxels with 1, 2, etc.

    # Fill holes in the combined volume
    filled_volume = correct_holes(combined_volume)  # Apply hole-filling using dominant phase

    # Validate the filled volume at interfaces
    filled_volume = validate_filled_volume(filled_volume, tomogram_volume, phase_ranges[folder])  # Correct interface voxels

    # Save the hole-filled combined volume
    print(f"Saving segmented hole-filled volume for {folder}")  # Log saving step
    for i, f in enumerate(slice_files):  # Iterate over slices
        cv2.imwrite(os.path.join(all_segment_folder, f), filled_volume[i])  # Save each slice as TIFF

    # Save individual phase masks
    print(f"Saving individual phase masks for {folder}")  # Log phase mask saving
    for phase, label in zip(sorted_phases, range(1, len(sorted_phases) + 1)):  # Iterate over phases and labels
        phase_volume = np.zeros_like(filled_volume)  # Initialize mask volume
        phase_volume[filled_volume == label] = 255  # Set phase voxels to 255
        phase_output_folder = os.path.join(phase_segment_folder, phase)  # Create phase-specific output folder
        os.makedirs(phase_output_folder, exist_ok=True)  # Ensure folder exists
        for i, f in enumerate(slice_files):  # Save each slice
            cv2.imwrite(os.path.join(phase_output_folder, f), phase_volume[i])  # Save phase mask as TIFF

    # Extract and save tomogram values for each phase
    print(f"Extracting and saving tomogram values for {folder}")  # Log extraction step
    for phase, label in zip(sorted_phases, range(1, len(sorted_phases) + 1)):  # Iterate over phases and labels
        extract_folder = os.path.join(extracted_folder, phase)  # Create phase-specific extraction folder
        os.makedirs(extract_folder, exist_ok=True)  # Ensure folder exists
        for i, f in enumerate(slice_files):  # Iterate over slices
            extracted_volume = np.zeros_like(tomogram_volume[i], dtype=np.uint16)  # Initialize empty slice (uint16)
            extracted_volume[filled_volume[i] == label] = tomogram_volume[i][filled_volume[i] == label]  # Copy tomogram intensities
            cv2.imwrite(os.path.join(extract_folder, f), extracted_volume)  # Save extracted slice as TIFF

print("✅ Processing complete.")  # Log completion of all processing