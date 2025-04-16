"""
Code Description:
This script applies K-Means clustering to 3D tomogram TIFF stacks to group non-zero voxel intensities, using masks to focus on regions of interest. It:
1. Recursively traverses a base directory to process tomogram and mask TIFF files in subfolders.
2. Applies K-Means clustering to non-zero voxel values, assigning each voxel the mean intensity of its cluster.
3. Saves the clustered tomograms in a mirrored output directory structure, preserving 16-bit data type.
Key features:
- Supports customizable clustering parameters (number of clusters, iterations, tolerance, seed) via command-line arguments.
- Filters out zero-valued voxels to optimize clustering on meaningful data.
- Ensures mask alignment with tomograms for accurate region selection.
Prerequisites:
- Input TIFF files must be readable by imageio and represent single-band image data.
Dependencies: os, numpy, imageio, scikit-learn, argparse (install via pip install numpy imageio scikit-learn).

Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, Dec 2024.
"""

import os  # Import os module for file and directory operations
import numpy as np  # Import NumPy for numerical operations and array handling
import imageio.v2 as imageio  # Import imageio.v2 for reading and writing TIFF files
from sklearn.cluster import KMeans  # Import KMeans for clustering voxel intensities
import argparse  # Import argparse for parsing command-line arguments

# Define base directories for tomograms, masks, and output clustered tomograms
tomogram_base_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\3-MaskedTomograms\Sw2"  # Directory for input tomogram TIFFs
mask_base_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\2-MaskTomograms\3-NonsolidMask\Sw2"  # Directory for mask TIFFs
clustered_output_dir = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\4-NonsolidClusteredTomograms\Sw2"  # Output directory for clustered tomograms

# Parse input arguments for flexibility in clustering parameters
parser = argparse.ArgumentParser(description="Perform K-Means clustering on 3D tomogram volumes.")  # Initialize argument parser with description
parser.add_argument("--n_clusters", type=int, default=9, help="Number of clusters for K-Means.")  # Add number of clusters argument
parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations for K-Means.")  # Add max iterations argument
parser.add_argument("--tol", type=float, default=1e-4, help="Tolerance for convergence of K-Means.")  # Add convergence tolerance argument
parser.add_argument("--seed", type=int, default=42, help="Random state for reproducibility of K-Means.")  # Add random seed argument
args = parser.parse_args()  # Parse the provided arguments

# Ensure the output directory exists
os.makedirs(clustered_output_dir, exist_ok=True)  # Create output base directory if it doesn’t exist

print("Starting tomogram clustering:")  # Log the start of clustering process
# Loop through each folder in the tomogram directory
for folder_name in os.listdir(tomogram_base_dir):  # Iterate over items in the tomogram base directory
    print(f"Processing folder: {folder_name}")  # Log the current folder being processed
    tomogram_folder = os.path.join(tomogram_base_dir, folder_name)  # Construct path to tomogram subfolder
    mask_folder = os.path.join(mask_base_dir, folder_name)  # Construct path to corresponding mask subfolder
    output_folder = os.path.join(clustered_output_dir, folder_name)  # Construct path for output clustered tomograms

    # Ensure the folder is valid and exists in both directories
    if os.path.isdir(tomogram_folder) and os.path.isdir(mask_folder):  # Verify both folders exist and are directories
        os.makedirs(output_folder, exist_ok=True)  # Create output subfolder if it doesn’t exist

        # Load the full 3D stack of tomograms and masks
        stack = []  # Initialize empty list to store masked slices
        file_names = sorted([f for f in os.listdir(tomogram_folder) if f.endswith('.tiff') or f.endswith('.tif')])  # Get sorted list of TIFF files
        for file_name in file_names:  # Iterate over TIFF files
            tomogram_path = os.path.join(tomogram_folder, file_name)  # Construct path to tomogram file
            mask_path = os.path.join(mask_folder, file_name)  # Construct path to corresponding mask file

            if os.path.exists(mask_path):  # Check if the mask file exists
                tomogram_slice = imageio.imread(tomogram_path).astype(np.uint16)  # Load tomogram slice as 16-bit
                mask_slice = imageio.imread(mask_path).astype(np.uint8)  # Load mask slice as 8-bit
                mask_slice = (mask_slice == 255).astype(np.uint16)  # Convert mask to binary (1 for 255, 0 elsewhere) as 16-bit
                masked_slice = tomogram_slice * mask_slice  # Apply mask to tomogram slice via element-wise multiplication
                stack.append(masked_slice)  # Append masked slice to the stack

        # Convert the stack into a 3D numpy array
        stack = np.stack(stack, axis=0)  # Stack masked slices into a 3D array (depth, height, width)

        # Flatten the 3D stack and filter out zero values
        flattened = stack.flatten()  # Flatten the 3D stack into a 1D array
        non_zero_indices = flattened > 0  # Identify indices of non-zero values
        non_zero_values = flattened[non_zero_indices]  # Extract non-zero values for clustering

        # Perform K-Means clustering on non-zero values
        kmeans = KMeans(  # Initialize KMeans clustering model
            n_clusters=args.n_clusters,  # Set number of clusters from arguments
            max_iter=args.max_iter,  # Set maximum iterations from arguments
            tol=args.tol,  # Set convergence tolerance from arguments
            random_state=args.seed  # Set random seed for reproducibility
        )
        clusters = kmeans.fit_predict(non_zero_values.reshape(-1, 1))  # Fit model and predict cluster labels

        # Compute the mean intensity value for each cluster
        cluster_means = np.zeros(args.n_clusters, dtype=np.uint16)  # Initialize array for cluster means
        for cluster_idx in range(args.n_clusters):  # Iterate over cluster indices
            cluster_means[cluster_idx] = np.mean(non_zero_values[clusters == cluster_idx]).astype(np.uint16)  # Compute mean intensity per cluster

        # Create a blank array for clustered output and assign mean intensity values
        clustered = np.zeros_like(flattened, dtype=np.uint16)  # Initialize output array with same size as flattened stack
        clustered[non_zero_indices] = [
            cluster_means[cluster_label] for cluster_label in clusters  # Assign mean intensity to each non-zero voxel
        ]

        # Reshape clustered array back to the original stack shape
        clustered_stack = clustered.reshape(stack.shape)  # Reshape 1D array to original 3D shape

        # Save each slice of the clustered stack as 16-bit images
        for i, file_name in enumerate(file_names):  # Iterate over file names and indices
            clustered_slice = clustered_stack[i]  # Extract the i-th clustered slice
            output_path = os.path.join(output_folder, file_name)  # Construct output file path
            imageio.imwrite(output_path, clustered_slice)  # Save clustered slice as TIFF

        print(f"Clustering complete and saved for {folder_name}.")  # Log completion for the folder

print("✅ Processing complete.")  # Log overall completion of processing