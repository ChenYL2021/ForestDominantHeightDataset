# ForestDominantHeightDataset
Mapping forest dominant height of China’s forests by UAV-LiDAR data

本代码用于地理空间 / 点云类数据的预处理、空间建模与可视化分析，覆盖数据尺度转换、坐标处理、批量格式化、空间插值、机器学习建模等流程，适用于遥感、LiDAR（激光雷达）、冠层高度模型（CHM）等领域的数据处理任务。
文件模块分类及功能说明
1. 数据预处理模块（DataPreprocess_*.py）
核心用于数据格式转换、坐标补充、批量处理及可视化预览：
DataPreprocess_1km_to_30m.py：实现数据空间分辨率从 1km 到 30m 的尺度转换
DataPreprocess_AddXY.py：为数据集添加空间 XY 坐标信息
DataPreprocess_BatchCSV.py：批量处理 CSV 格式数据集（如格式统一、字段清洗）
DataPreprocess_CHM30m.py/DataPreprocess_CHN_h.py：针对 ** 冠层高度模型（CHM）** 的专项预处理
DataPreprocess_PLOTLiDAR.py/DataPreprocess_PLOTData_pysldf.py：LiDAR 数据 / 通用数据集的可视化预览脚本
DataPreprocess_tifGetXY.py：从 TIFF 格式文件中提取空间坐标信息
2. 建模模块（Modelling_*.py）
实现空间插值与机器学习模型构建：
Modelling_HT_Kriging.py/Modelling_HT_twoKriging.py：基于 ** 克里金（Kriging）** 的空间插值建模（含单 / 双克里金方法）
Modelling_HT_Weight.py：带权重约束的空间建模脚本
Modelling_MultipleMachineModels.py：多机器学习模型的训练、评估与对比脚本
3. 辅助模块
Figures.py：分析结果的可视化脚本（生成图表、可视化报告）
Mapping_preHT.py：30m优势高地图生成
