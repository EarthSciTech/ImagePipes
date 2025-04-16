"""
 Code Description:
 This script analyzes 3D TIFF stacks of concentration data to compute a wide range of statistical and spatial 
 metrics across slices. It:
 1. Recursively processes TIFF files in subfolders of an input directory.
 2. Calculates metrics such as mean concentration, spreading, centroid/mode, skewness/kurtosis, axial mixing, 
    gradients, CHI, iso-surface area, and transverse/longitudinal variances.
 3. Saves results in Excel files with separate sheets for each metric, including global dispersion metrics, 
    with descriptive column names.
 Key features:
 - Applies a concentration threshold and normalizes data to a maximum value and the first slice.
 - Computes physical units (meters) using voxel size and handles edge cases (e.g., empty data).
 - Outputs detailed statistical summaries with means and standard deviations in a structured Excel format.
 Prerequisites:
 - Input TIFF files must be single-band concentration data readable by tifffile.
 Dependencies: os, numpy, tifffile, pandas (install via pip install tifffile pandas).

Code by: Amirsaman Rezaeyan, amirsaman.rezaeyan@gmail.com, Zürich, Switzerland, April 2025.
"""
import os  # Import os module for file and directory operations
import numpy as np  # Import numpy for numerical operations
import tifffile as tiff  # Import tifffile for reading TIFF files, aliased as tiff
import pandas as pd  # Import pandas for data handling and Excel output

# Input/output paths
input_path = r"Z:\users\ream\2-AnalysedData\3-InertiaEffectData\10-LiquidPhaseConcentration_basis"  # Define the input directory containing TIFF stacks
output_path = r"Z:\users\ream\2-AnalysedData\3-InertiaEffectData\18-Results\results_inertianalysis2"  # Define the output directory for Excel results

# Constants
voxel_size = 4.61e-6  # Define voxel size in meters (converted from micrometers: 4.61 µm)
voxel_area = voxel_size ** 2  # Calculate voxel area in square♥ meters (m²)
threshold = 0.2  # Set the concentration threshold for filtering data
max_concentration = 2.0  # Define the maximum concentration value for normalization

def process_folder(folder):  # Define a function to process a folder containing a TIFF stack
    tiff_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.tif')])  # Create a sorted list of TIFF files (case-insensitive)
    if not tiff_files:  # Check if there are no TIFF files in the folder
        return [None] * 13  # Return a list of 13 None values if no files are found (matches return structure)

    mean_concentration_data = []  # Initialize list for mean concentration data
    spreading_data = []  # Initialize list for spreading metrics
    centroid_mode_data = []  # Initialize list for centroid and mode data
    skew_kurt_data = []  # Initialize list for skewness and kurtosis data
    axial_mixing_data = []  # Initialize list for axial mixing metrics
    gradient_data = []  # Initialize list for gradient data
    chi_data = []  # Initialize list for CHI metric data
    iso_surface_data = []  # Initialize list for iso-surface area data
    transverse_variance_data = []  # Initialize list for transverse variance data
    longitudinal_variance_data = []  # Initialize list for longitudinal variance data
    centroid_z_positions = []  # Initialize list to store centroid z-positions
    total_mean_c, total_elevations = [], [] # Initialize global mean concentration and global elevation 
    trans_var_list, long_var_list = [], []  # Initialize lists for transverse and longitudinal variance values
    skew_list, kurt_list = [], []  # Initialize lists for skewness and kurtosis values
    mix_list, sdr_list = [], []  # Initialize lists for mixing scale and SDR values
    chi_list, iso_list = [], []  # Initialize lists for CHI and iso-surface area values

    first_slice_mean = None  # Initialize variable to store the mean of the first slice

    for i, fname in enumerate(tiff_files):  # Iterate over TIFF files with their indices
        slice_path = os.path.join(folder, fname)  # Construct the full path to the current TIFF file
        slice_data = tiff.imread(slice_path).astype(np.float32)  # Read the TIFF file into a NumPy array as 32-bit float
        elevation = i * voxel_size  # Calculate elevation in meters based on slice index and voxel size

        mask = slice_data >= threshold  # Create a boolean mask for values >= threshold
        data_thresh = slice_data[mask]  # Filter data to include only values above threshold
        
        # Mean Concentration
        mean_val = np.mean(data_thresh) if data_thresh.size > 0 else 0  # Calculate mean of thresholded data, 0 if empty
        std_val = np.std(data_thresh) if data_thresh.size > 0 else 0  # Calculate std dev of thresholded data, 0 if empty
        if first_slice_mean is None:  # Check if this is the first slice
            first_slice_mean = mean_val if mean_val > 0 else 1e-12  # Set first slice mean, use small value if 0 to avoid division by zero
        norm_max = mean_val / max_concentration  # Normalize mean to max concentration (2.0)
        std_norm_max = std_val / max_concentration  # Normalize std dev to max concentration
        norm_first = mean_val / first_slice_mean  # Normalize mean to first slice mean
        std_norm_first = std_val / first_slice_mean  # Normalize std dev to first slice mean
        mean_concentration_data.append([  # Append mean concentration metrics
            i, mean_val, std_val, norm_max, std_norm_max, norm_first, std_norm_first, elevation
        ])

        # Spreading
        # Spatial variance (σ²)
        # σ² = variance × area
        var = np.var(data_thresh) * data_thresh.size * voxel_area  # Calculate variance scaled by area and size
        std_var = np.std((data_thresh - np.mean(data_thresh))**2) * np.sqrt(data_thresh.size) * voxel_area  # Calculate std dev of variance
        
        # Cross-sectional Dilution index E
        # E = - ∑ C log(C) dx dy ≈ -∑ C log(C) * voxel_area (assuming normalised)
        flat = data_thresh[data_thresh > 0]  # Filter out zeros for entropy calculation
        C_norm = flat / np.sum(flat) if flat.sum() > 0 else np.array([1])  # Normalize concentrations, default to [1] if sum is 0
        entropy = -C_norm * np.log(C_norm)  # Compute entropy for each value
        E = np.sum(entropy) * data_thresh.size * voxel_area  # Calculate dilution index (E) scaled by area and size
        std_E = np.std(entropy) * np.sqrt(data_thresh.size) * voxel_area  # Calculate std dev of entropy
        
        spreading_data.append([i, var, std_var, elevation, E, std_E, elevation])  # Append spreading metrics

        # Centroid (xc = ∑x C(x) / ∑ C(x)) & Mode (maximum concentration position)
        if data_thresh.size > 0:  # Check if there are values above threshold
            coords = np.argwhere(mask)  # Get coordinates of thresholded pixels
            intensities = slice_data[mask]  # Get intensity values at those coordinates
            
            if intensities.size > 0:
                top_n = 10   # Get indices of top intensity values (e.g., concentration)
                top_indices = np.argpartition(intensities, -top_n)[-top_n:] if intensities.size >= top_n else np.arange(intensities.size)  # Get the indices of the top intensities
                top_coords = coords[top_indices] # Get the coordinates of the top intensities
                top_intensities = intensities[top_indices] # Get top intensities
                centroid = np.average(top_coords, axis=0, weights=top_intensities)  # Calculate weighted centroid (y, x)
            else:
                centroid = [0, 0]
                
            centroid_mode_data.append([i, centroid[1], centroid[0], elevation, *coords[np.argmax(intensities)][::-1], elevation])  # Append centroid (x, y) and mode (x, y)
        else:  # If no data above threshold
            centroid_mode_data.append([i, 0, 0, elevation, 0, 0, elevation])  # Append zeros for centroid and mode

        centroid_z_positions.append(elevation)  # Store elevation for longitudinal dispersion
        total_mean_c.append(mean_val) # Store total mean concentration
        total_elevations.append(elevation) # Store total elevation

        # Skewness E(C-C_mean)^3/std(C)^3 & Kurtosis E(C-C_mean)^4/std(C)^4
        if data_thresh.size > 2 and np.std(data_thresh) > 0:  # Check if enough data and non-zero std dev
            skew = np.mean(((data_thresh - np.mean(data_thresh)) / np.std(data_thresh))**3)  # Calculate skewness
            kurt = np.mean(((data_thresh - np.mean(data_thresh)) / np.std(data_thresh))**4)  # Calculate kurtosis
        else:  # If insufficient data or zero variance
            skew = kurt = 0  # Set skewness and kurtosis to 0
        skew_kurt_data.append([i, skew, 0, elevation, kurt, 0, elevation])  # Append with placeholder std dev (updated later)
        skew_list.append(skew)  # Store skewness for global std dev
        kurt_list.append(kurt)  # Store kurtosis for global std dev

        # Axial Mixing
        # Mixing scale: width of plume across transverse direction or sqrt of concentration variance
        mixing_scale = np.sqrt(np.var(data_thresh)) if data_thresh.size > 0 else 0  # Calculate mixing scale (sqrt of variance)
        
        # Scalar dissipation rate (SDR): 𝜒=𝐷∫∣∇𝐶∣^2 𝑑A normalised to D, which is reformulated as 𝜒/𝐷=∫∣∇𝐶∣^2 𝑑A [=] M^2.m^2
        grad_y, grad_x = np.gradient(slice_data)  # Compute gradients in y and x directions
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)  # Calculate gradient magnitude
        sdr = np.sum(grad_mag[mask]**2) * data_thresh.size * voxel_area  # Calculate Scalar dissipation rate (SDR) scaled by area
        axial_mixing_data.append([i, mixing_scale, 0, elevation, sdr, 0, elevation])  # Append with placeholder std dev
        mix_list.append(mixing_scale)  # Store mixing scale for global std dev
        sdr_list.append(sdr)  # Store SDR for global std dev

        # Gradient magnitude (integrated ‖∇C‖ across the domain)
        gradient_data.append([i, np.mean(grad_mag[mask]), np.std(grad_mag[mask]), elevation])  # Append mean and std dev of gradient magnitude

        # Concentration Homogeneity Index (CHI, 𝜁)
        # CHI = 1 - (σ²_C/σ²_Cmax)
        mean_c = np.mean(data_thresh) # Calculate mean concentration
        var_c = np.var(data_thresh) # Calculate mean concentration variance
        max_var = mean_c * (1 - mean_c) # Calculate maximum concentration variance
        chi = 1 - (var_c / max_var) if max_var > 0 else 0 # Calculate CHI metric
        chi_data.append([i, chi, 0, elevation])  # Append with placeholder std dev
        chi_list.append(chi)  # Store CHI for global std dev

        # Iso-Surface Area (it gives insight into plume fragmentation or mixing zones)
        iso_area = np.sum(mask) * voxel_area # Calculate iso-surface area as number of thresholded pixels times voxel area
        iso_surface_data.append([i, iso_area, 0, elevation])  # Append with placeholder std dev
        iso_list.append(iso_area)  # Store iso-surface area for global std dev

        # Transverse Variance
        if data_thresh.size > 0:  # Check if there are values above threshold
            coords = np.argwhere(mask)  # Get coordinates of thresholded pixels
            intensities = slice_data[mask]  # Get intensity values from slice_data where mask is True

            # Use top N concentration values to focus on the plume core
            top_m = 100  # Number of highest concentration points to consider
            top_indices = np.argpartition(intensities, -top_m)[-top_m:] if intensities.size >= top_m else np.arange(intensities.size) # Get the indices of the top intensities

            # Select top coordinates and corresponding intensities
            top_coords = coords[top_indices]  # Get top-N spatial coordinates
            top_intensities = intensities[top_indices]  # Get top-N concentration values

            # Extract x and y coordinates from top coordinates
            x = top_coords[:, 1]  # Extract x coordinates
            y = top_coords[:, 0]  # Extract y coordinates

            # Calculate weighted centroid (x_c, y_c) based on top concentrations
            x_c = np.average(x, weights=top_intensities)
            y_c = np.average(y, weights=top_intensities)

            # Compute weighted variance in x and y directions
            var_x = np.average((x - x_c)**2, weights=top_intensities)  # Variance in x direction
            var_y = np.average((y - y_c)**2, weights=top_intensities)  # Variance in y direction

            # Compute average transverse variance and scale it to physical units (m²)
            trans_var = ((var_x + var_y) / 2) * voxel_size**2
        else:  # If no data above threshold
            trans_var = 0  # Set transverse variance to 0

        transverse_variance_data.append([i, trans_var, 0, elevation])  # Append with placeholder std dev
        trans_var_list.append(trans_var)  # Store transverse variance for global std dev

        # Longitudinal Variance    
        long_var = (elevation - np.average(total_elevations, weights=total_mean_c))**2 if total_mean_c else 0  # Calculate longitudinal variance (simplified)
        longitudinal_variance_data.append([i, long_var, 0, elevation])  # Append with placeholder std dev
        long_var_list.append(long_var)  # Store longitudinal variance for global std dev

    # Global Std Dev Updates
    for row in skew_kurt_data: row[2] = np.std(skew_list); row[5] = np.std(kurt_list)  # Update std dev for skewness and kurtosis
    for row in axial_mixing_data: row[2] = np.std(mix_list); row[5] = np.std(sdr_list)  # Update std dev for mixing scale and SDR
    for row in chi_data: row[2] = np.std(chi_list)  # Update std dev for CHI☻
    for row in iso_surface_data: row[2] = np.std(iso_list)  # Update std dev for iso-surface area
    for row in transverse_variance_data: row[2] = np.std(trans_var_list)  # Update std dev for transverse variance
    for row in longitudinal_variance_data: row[2] = np.std(long_var_list)  # Update std dev for longitudinal variance

    # Global Dispersion Sheets
    # Transverse dispersion (D_T​ =1/N​ ∑σ_T^2), mean of slice-wise transverse variances.
    transverse_dispersion = [["Transverse Dispersion (D_T, m²)", np.mean(trans_var_list)],  # Calculate mean transverse dispersion
                             ["Std Dev", np.std(trans_var_list)]]  # Append std dev of transverse dispersion
    # Longitudinal dispersion (D_L = var(z_centroid)), variance of the z-centroid positions (across slices)
    longitudinal_dispersion = [["Longitudinal Dispersion (D_L, m²)", np.mean(long_var_list)],  # Calculate variance of z-positions
                               ["Std Dev", np.std(long_var_list)]]  # Append std dev of z-positions

    return (  # Return all computed data as a tuple
        mean_concentration_data, spreading_data, centroid_mode_data,
        skew_kurt_data, axial_mixing_data, gradient_data,
        chi_data, iso_surface_data,
        transverse_variance_data, longitudinal_variance_data,
        transverse_dispersion, longitudinal_dispersion
    )

def save_to_excel(filepath, *sheets):  # Define a function to save data to an Excel file with custom column names
    sheet_names = [  # Define sheet names for each data type
        "mean concentration", "spreading", "centroid & mode",
        "skewness & kurtosis", "axial mixing", "gradient magnitude",
        "CHI", "iso-surface area",
        "transverse variance", "longitudinal variance",
        "transverse dispersion", "longitudinal dispersion"
    ]
    with pd.ExcelWriter(filepath) as writer:  # Open an Excel writer object for the specified file path
        for sheet, name in zip(sheets, sheet_names):  # Iterate over data and corresponding sheet names
            if name == "mean concentration":  # Define columns for mean concentration sheet
                columns = ["Slice Number", "Mean (M)", "Std Dev (Mean, M)", "Norm to Max", "Std Dev (Norm to Max)", "Norm to First", "Std Dev (Norm to First)", "Elevation (m)"]
            elif name == "spreading":  # Define columns for spreading sheet
                columns = ["Slice Number", "Spatial Variance (σ², m²)", "Std Dev σ²", "Elevation (m)", "Dilution Index (E, m²)", "Std Dev E", "Elevation (m)"]
            elif name == "centroid & mode":  # Define columns for centroid & mode sheet
                columns = ["Slice Number", "Centroid X", "Centroid Y", "Elevation (m)", "Mode X", "Mode Y", "Elevation (m)"]
            elif name == "skewness & kurtosis":  # Define columns for skewness & kurtosis sheet
                columns = ["Slice Number", "Skewness", "std. dev. Skewness", "Elevation (m)", "Kurtosis", "std. dev. Kurtosis", "Elevation (m)"]
            elif name == "axial mixing":  # Define columns for axial mixing sheet
                columns = ["Slice Number", "Mixing Scale (m)", "std. dev. Mixing Scale (m)", "Elevation (m)", "SDR/D (M².m²)", "std. dev. SDR (M².m²)", "Elevation (m)"]
            elif name == "gradient magnitude":  # Define columns for gradient sheet
                columns = ["Slice Number", "Mean Gradient (M)", "Std Dev Mean Gradient (M)", "Elevation (m)"]
            elif name == "CHI":  # Define columns for CHI sheet
                columns = ["Slice Number", "CHI", "std. dev. CHI", "Elevation (m)"]
            elif name == "iso-surface area":  # Define columns for iso-surface area sheet
                columns = ["Slice Number", "Iso-Surface Area (m²)", "std. dev. Iso-Surface Area (m²)", "Elevation (m)"]
            elif name == "transverse variance":  # Define columns for transverse variance sheet
                columns = ["Slice Number", "Transverse Variance (m²)", "std. dev. Transverse Variance (m²)", "Elevation (m)"]
            elif name == "longitudinal variance":  # Define columns for longitudinal variance sheet
                columns = ["Slice Number", "Longitudinal Variance (m²)", "std. dev. Longitudinal Variance (m²)", "Elevation (m)"]
            elif name == "transverse dispersion, D_T":  # Define columns for transverse dispersion sheet
                columns = ["transverse dispersion (D_T, m²)", "Value"]
            elif name == "longitudinal dispersion D_L":  # Define columns for longitudinal dispersion sheet
                columns = ["longitudinal dispersion (D_L, m²)", "Value"]
            else:  # Fallback case (should not occur with current sheet names)
                columns = None
            pd.DataFrame(sheet, columns=columns).to_excel(writer, sheet_name=name, index=False)  # Write data to Excel with specified columns, no index

# Run script
for root, _, _ in os.walk(input_path):  # Use os.walk to recursively traverse the input directory
    folder_name = os.path.basename(root)  # Extract the name of the current folder
    if not folder_name:  # Check if the folder name is empty (e.g., root is input_path itself)
        continue  # Skip processing if no valid folder name
    print(f"Processing: {folder_name}")  # Print a message indicating processing has started
    results = process_folder(root)  # Process the folder and get all metrics
    if all(r is not None for r in results):  # Check if all results are valid (not None)
        os.makedirs(output_path, exist_ok=True)  # Create the output directory if it doesn’t exist
        excel_path = os.path.join(output_path, f"{folder_name}_analysis.xlsx")  # Construct the Excel file path
        save_to_excel(excel_path, *results)  # Save all results to the Excel file with custom column names
        print(f"{folder_name} saved.\n")  # Print a confirmation message

print("✅ Processing complete.")  # Print a final message when all processing is done
