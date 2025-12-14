# ForestDominantHeightDataset
Mapping forest dominant height of China’s forests by UAV-LiDAR data
##  1. Data Preprocessing Module (`DataPreprocess_*.py`)
This module handles **format conversion, spatial information completion, batch processing, and visualization preview**, serving as the foundation for modeling and mapping

- **`DataPreprocess_1km_to_30m.py`**
Spatial resolution scaling from **1 km to 30 m*
- **`DataPreprocess_AddXY.py`**
Adds spatial **X/Y coordinates** to sample datasets
- **`DataPreprocess_BatchCSV.py`**
Batch processing of CSV files (field cleaning, format normalization)
- **`DataPreprocess_CHM30m.py` / `DataPreprocess_CHN_h.py`**
Specialized preprocessing for **Canopy Height Model (CHM)** data
- **`DataPreprocess_PLOTLiDAR.py`**
Quick visualization and quality inspection for UAV-LiDAR / point cloud data
- **`DataPreprocess_PLOTData_pysldf.py`**
General-purpose visualization for spatial datasets
- **`DataPreprocess_tifGetXY.py`**
Extracts spatial coordinates from TIFF raster files


##  2. Modeling Module (`Modelling_*.py`)
This module focuses on **forest dominant height modeling and prediction**, combining classical geostatistics with machine learning approaches.

- **`Modelling_HT_Kriging.py`**
Spatial interpolation using **Ordinary Kriging**
- **`Modelling_HT_twoKriging.py`**
**Two-stage Kriging** for enhanced spatial modeling
- **`Modelling_HT_Weight.py`**
Weighted spatial modeling to improve prediction robustness
- **`Modelling_MultipleMachineModels.py`**
Training, evaluation, and comparison of multiple ML models (e.g., RF, XGBoost)

##  3. Mapping & Visualization

- **`Figures.py`**
Statistical analysis and visualization of modeling results

- **`Mapping_preHT.py`**
Generation of the final **30 m forest dominant height map**

## Application Scenarios
- Forest resource inventory and structural parameter mapping
- UAV-LiDAR and point cloud data analysis
- National and regional-scale forest structure products
- Integrated applications of **remote sensing + geostatistics + machine learning**


