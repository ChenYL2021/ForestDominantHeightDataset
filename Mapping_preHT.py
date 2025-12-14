# -*- coding: utf-8 -*-
# @Time    : 2024/1/6 20:39
# @Author  : ChenYuling
# @FileName: preH130M.py
# @Software: PyCharm
# @Describe：全区域跑30m

#%%
import arcpy
import os
from arcpy.sa import *
import pandas as pd
import numpy as np
from osgeo import gdal
import shutil
import joblib
from xgboost import XGBRegressor
#%忽略一些版本不兼容等警告
import warnings
warnings.filterwarnings("ignore")
#%%
# 输入的 Shapefile,在此确定是哪个植被区的范围，植被区
input_shapefile = r"F:\DB\TH\China_tile\alltile.shp"#TODO:修改处1*******************************************************
# 所有输入数据的tif
all_tiff_directory = r"G:\TH\Xdata30m\30m_forPredict\mask"#所有tiff位置,预测数据集
# 输出目录
shp_directory = r"G:\TH\Predict\zone\shp" #每个植被区下用于保存shp要素 #TODO:修改处2************************************************************
tif_directory = r"G:\TH\Predict\zone\tif" #每个植被区下用于保存每个要素下掩膜的tif#TODO:修改处3************************************************************
preTIF_directory = r"G:\TH\Predict\zone\preTIF" #每个植被区下用于保存预测的tif#TODO:修改处4************************************************************

# 创建输出目录（如果不存在）
if not os.path.exists(shp_directory):
    os.makedirs(shp_directory)

# 创建保存目录（如果不存在）
if not os.path.exists(tif_directory):
    os.makedirs(tif_directory)

# 创建保存目录（如果不存在）
if not os.path.exists(preTIF_directory):
    os.makedirs(preTIF_directory)
#%%
#最开始需要根据渔网格对其分大块，每个植被区只需运行一次就行
#TODO: 使用 SplitByAttributes工具分割 Shapefile
split_field = "TileName" #分割字段
arcpy.analysis.SplitByAttributes(input_shapefile, shp_directory, split_field)

#%%
#子函数，读取整个tiff像元值
def read_tiff_pixels(tiff_file):
    dataset = gdal.Open(tiff_file)
    band = dataset.GetRasterBand(1)  # 获取第一个波段（索引从1开始）
    # 读取整个波段的像元值
    pixel_data = band.ReadAsArray()
    # 输出像元值
    # print(pixel_data)
    dataset = None  # 关闭数据集
    return pixel_data


#子函数，清空文件中所有文件
def clear_folder(folder_path):
    # 确保文件夹存在
    if not os.path.exists(folder_path):
        return
    # 遍历文件夹中的文件并删除
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

#%%
#######step2 循环不同shapefile下不同tif（按切割块读信息）
# 获取文件夹中的tif文件路径列表
tiff_files = [file for file in os.listdir(all_tiff_directory) if file.endswith('.tif')]

for filename in os.listdir(shp_directory):
    if filename.endswith(".shp"):
        sub_shapefile = os.path.join(shp_directory, filename) #子shp：每一块
        sub_shapefile_name = filename.split('.')[0]  #获取该一块shp名称
        print('shp:',sub_shapefile_name)
        # 创建数据框，保存以子shp命名
        sub_shapefile_df = pd.DataFrame()#用该shp块进行掩膜所有tif输入数据
        # 清空文件夹
        clear_folder(tif_directory)
        # 循环遍历tif文件,获取某一小块的tif数据
        for tif_file in tiff_files:
            tif_path = os.path.join(all_tiff_directory, tif_file)
            tif_name = os.path.basename(tif_path).split('.')[0]  # 当前tif名称os.path.basename(file_path).split('.')[0]
            # 创建保存的tif名及位置
            output_filename = str(tif_name) + ".tif"
            output_tiff = os.path.join(tif_directory, output_filename)
            # TODO:执行按掩膜提
            print("Extract:", output_filename)
            arcpy.gp.ExtractByMask_sa(tif_path, sub_shapefile, output_tiff)#对应小块shp掩膜的所有tif保存临时目录在tif_directory文件里
        #
        #接下来读取掩膜获取的tif文件将其作为数据变量
        # 获取文件夹中的tif文件路径列表
        sub_tiff_files = [file for file in os.listdir(tif_directory) if file.endswith('.tif')]
        for subtif_file in sub_tiff_files:#该循环结束后，sub_shapefile_df数据框保存该小shp块下对应自变量列数据
            subtif_path = os.path.join(tif_directory, subtif_file)
            subtif_name = os.path.basename(subtif_path).split('.')[0]  # 当前tif名称os.path.basename(file_path).split('.')[0]
            print("Read:",subtif_name)
            tif_pixel_data = read_tiff_pixels(subtif_path)  # 读取整个tiff像元值,保存二维数组行列号
            ### TODO: 读取tif转为一维数组
            block_data_row = tif_pixel_data.flatten()  # 转一维数组
            sub_shapefile_df[subtif_name] = block_data_row  # 新读取tif数据作为数据框一列

        ####TODO: 对子自变量数据框进行处理，作为预测数据。主要3方面：NoData、变量顺序、土壤名称和土壤质地的哑变量化
        sub_shapefile_df['ID'] = range(len(sub_shapefile_df)) #新增自增列
        # 遍历所有列进行判断，取出值不等于NoData_value的子集
        for column in sub_shapefile_df.columns:
            sub_shapefile_df[column] = sub_shapefile_df[column].replace(-340282346638528859811704183484516925440.00000, np.nan)  #将所有列中NoData代表的最大值替换为空值
        #其他变量空值替换
        sub_shapefile_df['slope'] = sub_shapefile_df['slope'].replace(-128, np.NaN)  #将NoData代表的最大值替换为空值
        sub_shapefile_df['elevation'] = sub_shapefile_df['elevation'].replace(65535, np.NaN)  #将NoData代表的最大值替换为空值
        sub_shapefile_df['TRZD'] = sub_shapefile_df['TRZD'].replace(15, np.NaN)  #将NoData代表的最大值替换为空值
        sub_shapefile_df['TRMC'] = sub_shapefile_df['TRMC'].replace(-128, np.NaN)  #将NoData代表的最大值替换为空值,trmc
        sub_shapefile_df['pnf'] = sub_shapefile_df['pnf'].replace(127, np.NaN)  # 将NoData代表的最大值替换为空值
        #根据天然林和人工林pnf数据，选出森林需要预测的数据
        subset_pre = sub_shapefile_df[(sub_shapefile_df['pnf'] >0 ) & (sub_shapefile_df['pnf'] < 3)]
        #非森林区域，不需要预测
        subset_null = sub_shapefile_df[(sub_shapefile_df['pnf'] <=0 ) | (sub_shapefile_df['pnf'] >= 3)]

        #查看哪个字段需要NoData插值
        for column in subset_pre.columns:
            sub_cloumn = subset_pre[subset_pre[[column]].isnull().T.any()]
            if(len(sub_cloumn)>0):
                print(column," has NoData!")
                # TODO: 进行插值处理数据，保证数据不为空
                #  NoData_process()
            else:
                # TODO: 预测处理
                print(column," has not NoData")
        '''以下代码可以在ArcGIS Pro运行实现
        #%%TODO: 以上步骤若是检测到区域包含有空值，需要对对应区域文件中的对应tif属性文件进行空值插值，，其插值方式如下：替换3处
        #若是以上没有保证土壤名称trmc没有空值了，对插值的trmc*重新命名trmc（可在arcgis pro实现），其他过程变量移走
        output_raster_i = arcpy.sa.RasterCalculator(
            'Con(IsNull("trmc9.tif"), FocalStatistics("trmc9.tif", NbrRectangle(100,100, "CELL"), "MAJORITY"), "trmc9.tif")');
        output_raster_i.save(all_tiff_directory)
        '''

        # 重命名列名
        #调整数据与训练数据一致
        subset_pre0 = subset_pre[['aspect', 'bio1', 'bio4','bio12','bio15', 'elevation', 'NTHL', 'pnf', 'slope','TRHD', 'TRMC', 'TRZD', 'VH', 'VV','NDVI_MAX']]

        # %对于土壤名称和土壤质地取值情况问题
        # 读取 trmc属性表CSV 文件，获取其土壤名称
        trmc_csv_df = pd.read_csv('./DATA/TRMC_TABLE.csv')
        # 左连接两个数据框,添加土壤名称TRMC
        subset_pre1 = pd.merge(subset_pre0, trmc_csv_df, on='TRMC', how='left')
        # 删除trmc字段
        subset_pre2 = subset_pre1.drop('TRMC', axis=1)  # 保留soil

        # % 读取 trZD属性表CSV 文件，获取其土壤质地
        trzd_csv_df = pd.read_csv('./DATA/TRZD_TABLE.csv')
        # 左连接两个数据框,添加土壤名称TRZD
        subset_pre3 = pd.merge(subset_pre2, trzd_csv_df, on='TRZD', how='left')
        # 删除trmc字段
        subset_pre4 = subset_pre3.drop('TRZD', axis=1)  # 保留D_TEX
        # %处理离散数据-哑变量处理
        # 将列转换为字符类型
        subset_pre4['Soil'] = subset_pre4['Soil'].astype(str)
        subset_pre4['D_TEX'] = subset_pre4['D_TEX'].astype(str)

        subset_pre5 = pd.get_dummies(
            subset_pre4,
            columns=['Soil', 'D_TEX'],
            prefix=['TRMC', 'TRZD'],
            prefix_sep="_",
            dummy_na=False,
            drop_first=False
        )

        #%训练数据框列明
        train_columns = ['aspect', 'elevation', 'NTHL', 'slope', 'TCHD','bio1',  'bio4', 'bio12','bio15', 'VH', 'VV', 'NDVI_MAX', 'TRMC_ACf', 'TRMC_ACh',
                         'TRMC_ATc', 'TRMC_FRx', 'TRMC_LVj', 'TRMC_nan', 'TRZD_clay_heavy','TRZD_clay_light', 'TRZD_loam', 'TRZD_nan', 'TRZD_sandy_clay',
                         'TRZD_sandy_clay_loam', 'TRZD_sandy_loam', 'TRZD_silt','TRZD_silt_loam']

        for column in train_columns:#主要解决一些预测字段（土壤）不在训练字段中
            # 判断字符串是否在列表中
            if column in subset_pre5.columns:
                print(f"{column} 存在于训练特征表中")
            else:
                print(f"{column} 不存在于训练特征表中")
                subset_pre5[column] = 0
        #%
        subset_pre6 = subset_pre5[['aspect', 'elevation', 'NTHL', 'slope', 'TCHD','bio1', 'bio4', 'bio12', 'bio15', 'VH', 'VV', 'NDVI_MAX', 'TRMC_ACf', 'TRMC_ACh',
                         'TRMC_ATc', 'TRMC_FRx', 'TRMC_LVj', 'TRMC_nan', 'TRZD_clay_heavy','TRZD_clay_light', 'TRZD_loam', 'TRZD_nan', 'TRZD_sandy_clay',
                         'TRZD_sandy_clay_loam', 'TRZD_sandy_loam', 'TRZD_silt','TRZD_silt_loam']]
        #%
        import csv
        def save_array_to_csv(array, file_path):
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(array)

        #%将数据分成10份
        import math
        from tqdm import tqdm
        import joblib
        def cut_df(df, n):
            list_pre = []#创建空列表用于存储整个block预测值
            df_num = len(df)
            every_epoch_num = math.floor((df_num/n))
            for index in tqdm(range(n)):
                print(index)
                df_name = f'subset_pre6{index}'
                if index < n-1:
                    df_tem = df[every_epoch_num * index: every_epoch_num * (index + 1)]
                else:
                    df_tem = df[every_epoch_num * index:]
                df_name = df_tem
                df_name = df_name[['aspect', 'elevation', 'NTHL', 'slope', 'TCHD','bio1',  'bio4',  'bio12','bio15', 'VH', 'VV', 'NDVI_MAX', 'TRMC_ACf', 'TRMC_ACh',
                         'TRMC_ATc', 'TRMC_FRx', 'TRMC_LVj', 'TRMC_nan', 'TRZD_clay_heavy','TRZD_clay_light', 'TRZD_loam', 'TRZD_nan', 'TRZD_sandy_clay',
                         'TRZD_sandy_clay_loam', 'TRZD_sandy_loam', 'TRZD_silt','TRZD_silt_loam']]
                # df_name.to_csv("X_TRAIN.csv")
                print(len(df_name))
                pre_sub= df_name.values
                ###TODO:模型预测
                # 加载模型
                loaded_model1 = joblib.load("./DATA/THmodelA.model")
                # 使用模型对测试数据进行预测
                y_pred = loaded_model1.predict(pre_sub)
                # # 保存数组到CSV文件
                # save_array_to_csv(y_pred, 'y_pred.csv')
                y_pred_list = y_pred.tolist()
                list_pre= list_pre + y_pred_list
                print(y_pred)
            return list_pre
        pre_y_block = cut_df(subset_pre6, 1)#非空数据集预测结果
        #%将预测和空值数据合并在一起
        if len(subset_null) ==len(sub_shapefile_df):#预测全为空
            pre_2 = subset_null[['ID']]
            pre_2['preH1'] = np.NaN
            pre = pre_2
        else:
            pre_1 = subset_pre[['ID']]
            pre_1.loc[:,'preH1'] = pre_y_block
            pre_2 = subset_null[['ID']]
            pre_2['preH1'] = np.NaN
            pre = pd.concat([pre_1, pre_2], ignore_index=True)

        sorted_pre = pre.sort_values('ID')
        #%
        # ###TODO:将数组转换为预测小tif
        basepathtif = r"F:\DB\TH\Predict\zone\tif\pnf.tif" #TODO:修改处5************************************************************
        # 读取tif文件
        origin_dataset = gdal.Open(basepathtif)
        # 获取数据集的行数和列数(维度和numpy数组相反)
        rows = origin_dataset.RasterYSize
        cols = origin_dataset.RasterXSize
        ###TODO:处理像元值数据
        # 提取某一列的值作为一维数组
        preAGE_values = sorted_pre['preH1'].values
        # 将一维数组转换为二维数组
        preAGE_2d = np.reshape(preAGE_values, (rows, cols))
        #%
        # 创建保存的tif名及位置
        output_filename = str(sub_shapefile_name) + '_preH1.tif'
        # output_filename = 't1_preAGE.tif'
        output_tiff = os.path.join(preTIF_directory, output_filename)
        '''
        driver.Create(filename, xsize, ysize, [bands], [data_type], [options])
        filename是要创建的数据集的路径。
        xsize是新数据集中的列数。
        ysize是新数据集中的行数。
        bands是新数据集中的波段数。默认值为1。
        data_type是将存储在新数据集中的数据类型。默认值为GDT_Byte。
        options是创建选项字符串的列表。可能的值取决于正在创建的数据集的类型。
        '''
        # 创建tif文件
        driver = gdal.GetDriverByName('GTiff')
        # 维度和numpy数组相反
        new_dataset = driver.Create(output_tiff, preAGE_2d.shape[1], preAGE_2d.shape[0], 1, gdal.GDT_Float32)

        # 读取之前的tif信息，为新生成的tif添加地理信息
        # 如果不增加这一步，则生成的图片没有经纬度坐标、投影的信息
        # 获取投影信息
        origin_proj = origin_dataset.GetProjection()
        new_dataset.SetProjection(origin_proj)
        # 仿射矩阵
        origin_geotrans = origin_dataset.GetGeoTransform()
        new_dataset.SetGeoTransform(origin_geotrans)

        band = new_dataset.GetRasterBand(1)
        band.WriteArray(preAGE_2d)
        print(new_dataset)
        new_data = new_dataset.ReadAsArray()
        print(new_data)

        # 关闭数据集
        origin_dataset = None
        new_dataset = None
        print("保存成功！")

#%%###TODO:循环结束后，一个植被区所有子块都生成tif，将文件夹预测小tif进行合并形成一个植被区预测tif



