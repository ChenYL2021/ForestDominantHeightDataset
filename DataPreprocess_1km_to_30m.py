# -*- coding: UTF-8 -*-
# -*- coding: utf-8 -*-
# @Time    : 2024/10/1 21:42
# @Author  : ChenYuling
# @Software: PyCharm
# @Describe：数据处理部分，将大样地切成30*30m的小样方
import csv
import numpy
import numpy as np
import laspy
import math
import pandas as pd
import os
from osgeo import osr

def get_all_filenames(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            name_without_extension, _ = os.path.splitext(filename)
            filenames.append(name_without_extension)
    return filenames

#给出las文件夹的路径，csv文件夹的路径，默认las命名=csv+_转换为las,resolution给出对应的分辨率大小

def resolution_change(input_las_path,input_csv_path,name,resolution,output_plot_path,output_spatial_path,zone):
    print(name)
    data=pd.read_csv(input_csv_path+name+"_统计单木属性.csv",skiprows=0, encoding = 'gb2312')#
    # data = pd.read_csv(input_csv_path + name + ".csv", skiprows=0)  #, encoding='gb2312'
    header = data.columns.tolist()
    values=np.array(data.values.tolist())
    inFile = laspy.read(input_las_path+name+"_转换为Las.las")
    x=data[data.columns[1]].tolist() #Tree_ID对应的x
    y=data[data.columns[2]].tolist() #Tree_ID对应的y
    x_max, y_max = inFile.header.max[0:2]
    x_min, y_min = inFile.header.min[0:2]
    xOrigin = x_min
    yOrigin = y_max

    cols = int(math.ceil(((x_max - x_min) / resolution)))
    rows = int(math.ceil(((y_max - y_min) / abs(resolution))))
    xOffset = (x - xOrigin) / resolution
    xOffset = xOffset.astype(int)
    yOffset = (y - yOrigin) / -resolution
    yOffset = yOffset.astype(int)
    tick=0
    #第一个输出输出每个plot中的值
    #按照input_csv中的方式进行输出
    spatial_reference=[]
    if(cols<3 or rows <3):
        for i in range(0, cols):
            for j in range(0, rows):
                index = np.where((xOffset == i) & (yOffset == j))
                index = index[0]
                if (len(index) != 0):
                    data_now = values[index]
                    data_now = data_now.tolist()
                    df = pd.DataFrame(data_now, columns=header)
                    df.to_csv(output_plot_path + name + "_" + str(i) + "_" + str(j) + ".csv", index=False,
                              encoding='utf-8-sig')  #
                    outRasterSRS = osr.SpatialReference()
                    outRasterSRS.ImportFromProj4("+proj=utm +zone=" + str(zone) + " +datum=WGS84 +units=m +no_defs")
                    geosc1 = outRasterSRS.CloneGeogCS()
                    cor_tran = osr.CoordinateTransformation(outRasterSRS, geosc1)
                    coords = cor_tran.TransformPoint((i + 1 / 2) * resolution + xOrigin,
                                                     (j + 1 / 2) * (-resolution) + yOrigin)
                    # print(coords[0], coords[1])
                    spatial_reference.append(
                        [output_plot_path + name + "_" + str(i) + "_" + str(j) + ".csv", coords[0], coords[1], zone])
    else:#执行去边界操作
        for i in range(1,cols-1):
            for j in range(1,rows-1):
                index=np.where((xOffset==i) & (yOffset==j))
                index_left=np.where((xOffset==i) & (yOffset==j-1))
                index_right=np.where((xOffset==i) & (yOffset==j+1))
                if(len(index)!=0 and len(index_left)>0 and len(index_right)>0):
                    data_now=values[index]
                    data_now=data_now.tolist()
                    df = pd.DataFrame(data_now, columns=header)
                    df.to_csv(output_plot_path+name+"_"+str(i)+"_"+str(j)+".csv", index=False,encoding='utf-8-sig')#
                    outRasterSRS = osr.SpatialReference()
                    outRasterSRS.ImportFromProj4("+proj=utm +zone=" + str(zone) + " +datum=WGS84 +units=m +no_defs")
                    geosc1 = outRasterSRS.CloneGeogCS()
                    cor_tran = osr.CoordinateTransformation(outRasterSRS, geosc1)
                    coords = cor_tran.TransformPoint((i + 1 / 2) * resolution + xOrigin, (j + 1 / 2) * (-resolution) + yOrigin)
                    #print(coords[0], coords[1])
                    spatial_reference.append(
                        [name + "_" + str(i) + "_" + str(j), coords[0], coords[1], zone])


    df_new = pd.DataFrame(spatial_reference, columns=['plot','x','y','zone'])
    df_new.to_csv(output_spatial_path + name + "_plot_spatial.csv", index=False,encoding='utf-8-sig')



input_las_path = "G://lidarDATA//zone49//las49/"
input_csv_path = "G://lidarDATA//zone49//csv49/"
output_plot_path = "G://lidarDATA//zone49//plot100/"
output_spatial_path = "G://lidarDATA//zone49//spatial100/"
filenames=get_all_filenames(input_csv_path)
for name in filenames:
    subcsvname =  name.replace("_统计单木属性","")
    # subcsvname =  name
    resolution_change(input_las_path,input_csv_path,subcsvname,100,output_plot_path,output_spatial_path,49)

