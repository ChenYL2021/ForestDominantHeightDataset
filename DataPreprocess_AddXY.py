# -*- coding: utf-8 -*-
# @Time    : 2023/10/25 9:54
# @Author  : ChenYuling
# @FileName: AddXY.py
# @Software: PyCharm
# @Describe：数据处理部分：对渔网30*30的格子面图层属性表进行添加XY字段

# #%%单个文件实现
# import arcpy
# subShp = r"F:\DB\CHM\dongbeidayangdi\fishnet\dongbeidayangdi5_CHM.shp"   #注意文件文件夹符号为/
# #制定表空间#第一个是表名、第二个是字段名、第三个是字段类型、第四个是字段长度
# arcpy.AddField_management(subShp,"X","double")
# arcpy.AddField_management(subShp,"Y","double")

#%%批量实现
import os
import arcpy
SHP_directory = r"F:\DB\CHM\dongbeidayangdi\fishnet"

# 获取文件夹中的tif文件路径列表
sub_SHP_files = [file for file in os.listdir(SHP_directory) if file.endswith('.shp')]
for subshp_file in sub_SHP_files:
    subshp_path = os.path.join(SHP_directory, subshp_file)
    # 制定表空间#第一个是表名、第二个是字段名、第三个是字段类型、第四个是字段长度
    arcpy.DeleteField_management(subshp_path, "X")
    arcpy.DeleteField_management(subshp_path, "Y")
    arcpy.AddField_management(subshp_path, "PLOTX", "double")
    arcpy.AddField_management(subshp_path, "PLOTY", "double")



