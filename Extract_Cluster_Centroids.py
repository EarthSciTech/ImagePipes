"""
Code Description:
This script analyzes TIFF images in a specified folder to count non-zero voxel intensities (cluster centroids) and exports the results. It:
1. Reads all TIFF files in a given folder using Pillow.
2. Counts the occurrences of each non-zero greyscale value across all images.
3. Saves the greyscale values and their counts to an Excel file and the values alone to a text file.
Key features:
- Excludes zero-valued voxels to focus on meaningful intensities.
- Sorts greyscale values for consistent reporting.
- Outputs results in both Excel (value-count pairs) and text (values only) formats for flexibility.
Prerequisites:
- Input TIFF files must be readable by Pillow and represent single-band image data.
Dependencies: os, numpy, pillow, pandas (install via pip install pillow numpy pandas).

Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, Jan 2025.
"""

import os  # Import os module for file and directory operations
import numpy as np  # Import NumPy for numerical operations and array handling
from PIL import Image  # Import Pillow’s Image module for reading TIFF files
import pandas as pd  # Import pandas for creating and exporting Excel files

# Define the folder path containing the .tif files
folder_path = r"Y:\users\ream\2-DataAnalysis\4-ChaoticAdvection\Series1\4-NonsolidClusteredTomograms\Sw2"  # Specify the input folder containing TIFF images

# Check if the folder exists
if not os.path.exists(folder_path):  # Verify that the specified folder exists
    raise FileNotFoundError(f"The folder path does not exist: {folder_path}")  # Raise an error if folder is missing

# Initialize a dictionary to store voxel counts for each greyscale value (excluding zero)
voxel_counts = {}  # Create an empty dictionary to track non-zero greyscale values and their counts

# Loop through all files in the folder
for file_name in os.listdir(folder_path):  # Iterate over all files in the folder
    if file_name.endswith(".tif"):  # Check if the file has a .tif extension
        file_path = os.path.join(folder_path, file_name)  # Construct the full path to the TIFF file
        
        # Open the .tif file using Pillow
        with Image.open(file_path) as img:  # Open the TIFF file in a context manager
            # Convert the image to a numpy array for processing
            img_array = np.array(img)  # Convert the Pillow image to a NumPy array
            
            # Get unique values and their counts from the image array
            unique, counts = np.unique(img_array, return_counts=True)  # Compute unique values and their frequencies
            
            # Update the voxel_counts dictionary with non-zero values
            for value, count in zip(unique, counts):  # Iterate over unique values and their counts
                if value != 0:  # Exclude zero values (background or irrelevant)
                    if value in voxel_counts:  # Check if the value is already in the dictionary
                        voxel_counts[value] += count  # Increment the count for the existing value
                    else:  # If the value is encountered for the first time
                        voxel_counts[value] = count  # Initialize the count for the new value

# Convert the voxel_counts dictionary to a sorted list of tuples (sorted by greyscale value)
sorted_voxel_counts = sorted(voxel_counts.items())  # Sort the dictionary items by greyscale value

# Extract values and counts for reporting
values_only = [value for value, count in sorted_voxel_counts]  # Extract just the greyscale values
values_and_counts = sorted_voxel_counts  # Keep the full list of (value, count) pairs

# Output: Save values and counts in an Excel file
excel_output_file = os.path.join(folder_path, "1_centroids_counts.xlsx")  # Define path for the Excel output file
df = pd.DataFrame(values_and_counts, columns=["centroids Greyscale Value", "Voxel Count"])  # Create a DataFrame with value-count pairs
df.to_excel(excel_output_file, index=False)  # Save the DataFrame to Excel without row indices
print(f"Values and counts saved to Excel: {excel_output_file}")  # Log the Excel file path

# Output: Save values only in a text file
txt_output_file = os.path.join(folder_path, "2_centroids.txt")  # Define path for the text output file
with open(txt_output_file, "w") as f:  # Open the text file in write mode
    f.write(", ".join(map(str, values_only)))  # Write comma-separated greyscale values
print(f"Centroids Values only saved to text file: {txt_output_file}")  # Log the text file path

# Print values only
print("centroids Greyscale Values (Non-zero):")  # Print header for greyscale values
print(values_only)  # Print the list of non-zero greyscale values
print("✅ Processing complete.")  # Log completion of processing