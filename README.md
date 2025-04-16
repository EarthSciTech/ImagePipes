🧠 ImagePipes: MicroCT Image Processing Pipelines
ImagePipes is a modular and extensible Python-based toolkit designed for processing, analyzing, and segmenting high-resolution 3D micro-computed tomography (µCT) images. Developed with a focus on porous media and geoscientific applications, this suite of scripts enables streamlined, reproducible workflows for transforming raw µCT image data into meaningful quantitative insights.

These pipelines were created and tested in real-world research projects at ETH Zürich, Eawag, and Empa, and are now made publicly available to support the broader scientific and engineering community.

🔧 Core Functionalities

📌 Image Registration
Rigid and Affine Registration with Elastix
Align misaligned or distorted image stacks using robust multi-resolution, affine transformations via the Elastix library.

Phase Cross-Correlation Registration
Quickly compute and apply translation-only corrections using Fourier-based phase cross-correlation, optimized for small shifts in µCT stacks.

🧼 Noise Reduction
2D and 3D Non-Local Means (NLM) Denoising
Apply structure-preserving NLM filters on a slice-by-slice or volumetric basis, tuned to match Avizo/Amira denoising settings. Handles various data types and supports zero-padding.

🧊 Segmentation & Clustering
K-Means Clustering of Masked Tomograms
Segment tomograms into clusters based on voxel intensity using scikit-learn’s KMeans. Customizable parameters allow flexible unsupervised classification.

Phase Segmentation
Post-process clustered tomograms to assign phase labels (e.g., solid, liquid, air). Includes hole-filling using neighborhood filtering and interface validation against raw intensity values.

✂️ Edge Enhancement
3D Binary Erosion-Based Edge Refinement
Improve segmentation quality near phase interfaces by applying morphological erosion in 3D, with special handling for boundary slices.

🔄 Conversion Tools
CT Number to Concentration Conversion
Linearly map CT grayscale intensities to physical concentration values using experimentally calibrated slope/intercept values. Includes thresholding and metadata preservation.

3D Mesh Generation (Marching Cubes)
Generate watertight surface meshes (.PLY format) from binary image stacks using marching cubes, with control over voxel spacing for real-world scale reconstruction.

🔁 Resampling & Masking
3D Isotropic Resampling
Rescale volumetric image data to desired voxel sizes using nearest-neighbor or other interpolation methods, preserving binary or grayscale fidelity.

Tomogram Masking
Apply binary masks to extract regions of interest across large tomogram stacks. Supports folder hierarchies and data type consistency.

📂 Use Cases
ImagePipes was built for high-resolution µCT analysis in:

Reactive transport modeling

Pore-scale flow visualization

Multiphase fluid characterization

CO₂ trapping and leakage risk analysis

Fracture-matrix interface studies

Sediment structure and grain morphology assessment

Its modular structure also makes it useful in materials science, medical imaging, and machine learning pre-processing.

🧰 Built With
NumPy – numerical computation

SciPy – advanced image processing and resampling

scikit-image – denoising, registration, and morphology

scikit-learn – clustering

SimpleITK – medical image registration and format conversion

OpenCV – image I/O and processing

trimesh – mesh generation and export

tifffile – TIFF read/write

rasterio – geospatial TIFF transformation

🔓 License
This project is licensed under the MIT License – see the LICENSE file for details.
You are free to use, modify, distribute, and build upon this work with attribution.

👨‍🔬 Author
Amirsaman Rezaeyan
Researcher at ETH Zürich, Eawag, & Empa
📍 Zürich, Switzerland
📧 amirsaman[dot]rezaeyan[@]gmail.com
