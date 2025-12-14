# -*- coding: utf-8 -*-
# @Time    : 2023/12/1 13:15
# @Author  : ChenYuling
# @FileName: PLOTlidar.py
# @Software: PyCharm
# @Describe：数据处理部分：关于整理的lidar的1km数据集，读取las数据，获取中心点坐标（WGS84）,计算样地获取的树高一系列变量
###############################################################################################################################
#第1阶段，对las数据批量处理提取样地plot数据中坐标
###############################################################################################################################
import os
import laspy
import numpy as np
from scipy import spatial
import pandas as pd
import open3d as o3d
import math
from osgeo import osr
from scipy.interpolate import UnivariateSpline, CubicSpline
from osgeo import gdal

def get_all_filenames(folder_path):
    filenames = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            name_without_extension, _ = os.path.splitext(filename)
            filenames.append(name_without_extension)
    return filenames


def circle_fit(A, B, C):
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    if D == 0:
        return (0, 0), 0

    Ux = (np.dot(A, A) * (B[1] - C[1]) + np.dot(B, B) * (C[1] - A[1]) + np.dot(C, C) * (A[1] - B[1])) / D
    Uy = (np.dot(A, A) * (C[0] - B[0]) + np.dot(B, B) * (A[0] - C[0]) + np.dot(C, C) * (B[0] - A[0])) / D
    center = (Ux, Uy)
    radius = np.linalg.norm(np.array(A) - np.array(center))
    return center, radius


def fit_func(x, a, b, d):
    return np.sqrt(np.average((x[0] - a) ** 2 + (x[1] - b) ** 2)) - d

def welzl(points):
    tick = 0  # 指示递归次数

    def welzl_helper(R, P, R_size, tick, center=(0, 0), radius=0):
        if R_size == 3 or not P:
            if R_size == 0:
                return center, radius

            if R_size == 1:
                center = R[0]
                return center, radius

            if R_size == 2:
                center = ((R[0][0] + R[1][0]) / 2, (R[0][1] + R[1][1]) / 2)
                radius = np.linalg.norm(np.array(R[0]) - np.array(center))
                return center, radius

            if R_size == 3:
                A, B, C = R
                center, radius = circle_fit(A, B, C)
                return center, radius

            else:
                x = np.array([point[0] for point in R])
                y = np.array([point[1] for point in R])

                # 使用最小二乘法进行圆拟合
                A = np.vstack([2 * x, 2 * y, np.ones_like(x)]).T
                B = x ** 2 + y ** 2
                m = np.linalg.lstsq(A, B, rcond=None)[0]

                # 提取拟合的圆参数
                center = m[:2]
                radius = np.sqrt(m[2] + np.dot(center, center))
                return center, radius
        tick = tick + 1
        # print(tick)

        rand_index = np.random.randint(0, len(P))
        P[rand_index], P[-1] = P[-1], P[rand_index]
        center, radius = welzl_helper(R, P[:-1], R_size, tick, center, radius)
        if np.linalg.norm(np.array(P[-1]) - np.array(center)) <= radius:
            return center, radius
        else:
            R.append(P[-1])
            return welzl_helper(R, P[:-1], len(R), tick)

    if len(points) <= 2:
        if len(points) == 0:
            return (0, 0), 0
        elif len(points) == 1:
            return points[0], 0
        else:
            return welzl_helper([points[0]], [points[1]], 1, tick)

    P = list(points)
    np.random.shuffle(P)
    return welzl_helper([P[0]], P[1:], 1, tick)

# 输入单棵树的点云，然后对其做大小为d的间隔点云
def get_CBH(x, y, z, d):
    z_min = min(z)
    z_max = max(z)
    z_range = np.arange(0, z_max + d, d)
    number = np.zeros(len(z_range))
    # 先构建点云分布曲线
    k = np.where(z > -1)

    for i in range(len(z)):
        number[int(np.floor((z[i]) / d))] += 1

    spline = CubicSpline(z_range, number)
    derivative = spline.derivative()
    # 二阶导数
    derivative_two = derivative.derivative()
    # 去除林下点云
    x_values = np.linspace(z_max, 0, 10000)  # 生成一些评估点
    zero_points = []
    density = []
    low_point = np.array([])

    last_one = x_values[0]
    for i in x_values:
        if (last_one * derivative_two(i) < 0):
            root = find_root(derivative_two, last_one, i)
            zero_points.append(root)
        last_one = i
    for i in range(len(zero_points) - 1):
        if (derivative((zero_points[i] + zero_points[i + 1]) / 2) > 0):
            flag = np.where(z < zero_points[i])
            flag2 = np.where(z > zero_points[i + 1])
            flag = np.intersect1d(flag, flag2)
            density.append(len(z[flag]) / (zero_points[i] - zero_points[i + 1]))
            low_point = np.append(low_point, zero_points[i + 1])
    # 筛选掉高于50%的点
    if (len(density) > 0):
        sort_index = np.argsort(density)
        count = 0
        p = low_point[sort_index[count]]
        while (p > 0.5 * z_max and count < len(sort_index)):
            p = low_point[sort_index[count + 1]]
            count = count + 1
        if (count == len(sort_index)):
            p = -1
        # print(p)
        k = np.where(z > p)
        z = z[k]
        x = x[k]
        y = y[k]
    '''
    '''
    x_utilized = np.average(x)
    y_utilized = np.average(y)
    x = x - x_utilized
    y = y - y_utilized
    z_min = min(z)
    z_max = max(z)
    z_range = np.arange(z_min, z_max + d, d)
    number_ori = np.zeros(len(z_range))
    number_new = np.zeros(len(z_range))

    for i in range(len(z)):
        number_ori[int(np.floor((z[i] - z_min) / d)):] += 1
    # 重新构建点云分布曲线
    # 按照上层及下层密度进行补充或者做外轮廓曲线？

    # region 改动位置
    # 找到高于50百分数的点云密度最大位置

    for i in range(len(z)):
        number_new[int(np.floor((z[i] - z_min) / d))] += 1

    # 这里还是找的点最多的位置，先看看结果
    sort_index_number_z = np.argsort(number_new)
    z_maximum_density = sort_index_number_z[-1] * 0.1 + 0.1 + z_min

    if (z_maximum_density > 0 and number_new[sort_index_number_z[-1]] > 5 and number_new[sort_index_number_z[-1]] < 1000):
        flag = np.where(z > z_maximum_density - 0.1)
        flag2 = np.where(z <= z_maximum_density)
        index = np.intersect1d(flag, flag2)
        x_maximum_density = x[index]
        y_maximum_density = y[index]
        points_2D = np.column_stack((x_maximum_density, y_maximum_density))

        center, radius = welzl(points_2D)
        density_standard = number_new[sort_index_number_z[-1]] / (radius * radius)
        x_center = np.average(x)
        y_center = np.average(y)  # 中心点

        for i in range(len(number_new)):
            # 考虑到一些基本没有点的位置，可能是一些遮挡或者其他位置的点？那我认为单木分割的结果是准确的，因此能够检测到最下层的位置
            flag = np.where(z > i * 0.1 + z_min)
            flag2 = np.where(z < i * 0.1 + z_min + 0.1)
            index = np.intersect1d(flag, flag2)

            x_now = x[index]
            y_now = y[index]
            if (len(x_now) > 1):
                r = np.max(np.sqrt(np.square(x_now - x_center) + np.square(y_now - y_center)))
            else:
                r = np.sqrt(np.square(x_now - x_center) + np.square(y_now - y_center))

            if (number_new[i] / (r * r) < density_standard):
                number_new[i] = int(density_standard * r * r)
    for i in range(len(number_new) - 1):
        number_new[i + 1] = number_new[i] + number_new[i + 1]
        # 每个位置潜在的中心点
        # end region 改动位置

    spline_ori = CubicSpline(z_range, number_ori)
    spline_new = CubicSpline(z_range, number_new)
    derivative_ori = spline_ori.derivative()
    derivative_two_ori = derivative_ori.derivative()
    derivative_new = spline_new.derivative()
    derivative_two_new = derivative_new.derivative()
    '''
    x_values = np.linspace(z_min, z_max, 10000)
    y_ori=spline_ori(x_values)
    y_new=spline_new(x_values)
    y_d_1_ori = derivative_ori(x_values)
    y_d_1_new =derivative_new(x_values)
    y_d_2_ori = derivative_two_ori(x_values)
    y_d_2_new = derivative_two_new(x_values)
    plt.plot(z_range, number_ori, 'o', label='Original Data')
    plt.plot(x_values, y_ori, label='Ori 0')
    plt.plot(x_values, y_new, label='New 0')
    plt.plot(x_values,  y_d_1_ori, label='Ori 1')
    plt.plot(x_values, y_d_1_new, label='New 1')
    plt.plot(x_values, y_d_2_ori, label='Ori 2')
    plt.plot(x_values, y_d_2_new, label='New 2')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Cubic Spline Interpolation')
    plt.grid(True)
    plt.show()
    '''

    spline = spline_new
    derivative = spline.derivative()
    # 二阶导数
    derivative_two = derivative.derivative()
    # 单木的CBH
    x_values = np.linspace(z_max, z_min, 10000)  # 生成一些评估点
    max_derivative1_point = None
    root_jihe = []
    last_one = x_values[0]
    for p in x_values:
        if (derivative_two(last_one) * derivative_two(p) < 0):
            root = find_root(derivative_two, last_one, p)
            '''
            if (max_derivative1_point == None or derivative(root) > derivative(max_derivative1_point)):
                if (root < 0.5 * z_max):
                    max_derivative1_point=root
        last_one = p
        '''
            if (root < 0.5 * z_max and root > 1):
                # max_derivative1_point = root
                root_jihe.append(root)
        last_one = p

    if (len(root_jihe) > 1):
        deri = np.array([derivative(point) for point in root_jihe])
        iu = np.argsort(deri)
        if (iu[-1] < len(root_jihe) - 1):
            max_derivative1_point = root_jihe[iu[-1] + 1]
        else:
            max_derivative1_point = root_jihe[iu[-1]]
    else:
        if (len(root_jihe) == 1):
            max_derivative1_point = root_jihe[0]

    return max_derivative1_point

def find_root(func, a, b, tol=1e-6, max_iter=10):
    """
    使用二分法查找函数的零点。

    参数：
    func (callable) - 要查找零点的函数。
    a (float) - 搜索区间的左边界。
    b (float) - 搜索区间的右边界。
    tol (float) - 允许的误差容限。
    max_iter (int) - 最大迭代次数。

    返回：
    root (float) - 找到的零点。
    iterations (int) - 迭代次数。
    """

    root = None
    iterations = 0

    while (b - a) / 2.0 > tol and iterations < max_iter:
        midpoint = (a + b) / 2.0
        if func(midpoint) == 0:
            root = midpoint
            break
        elif func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint
        iterations += 1

    if root is None:
        root = (a + b) / 2.0

    return root

def las_CBH_create(las_file_name, output_file_name):
    # las文件位置 示例数据为机载数据 但是算法通用
    lasfile = las_file_name
    # 打开las文件
    inFile = laspy.read(lasfile)

    tree_id = inFile.Tree_ID
    x, y, z = inFile.x, inFile.y, inFile.z
    d = 0.1
    # 提取边界

    CBH_average = 0
    tick = 0
    CBH_max = 0
    CBH_min = 100
    id_min=min(tree_id[tree_id>0])
    id_max = max(tree_id)
    CBH = []
    for i in range(int(id_min-1),int(id_max)):
        if (np.sum(tree_id == (i + 1)) > 100):
            x_used = x[tree_id == (i + 1)]
            y_used = y[tree_id == (i + 1)]
            z_now = z[tree_id == (i + 1)]
            CBH_now = get_CBH(x_used, y_used, z_now, d)
            if(CBH_now!=None):
                CBH.append((i + 1, CBH_now))
                print(i+1)
                print(CBH_now)
    df = pd.DataFrame(CBH, columns=['TreeID', 'CBH'])
    df.to_csv(output_file_name, index=False)



#%%
# 存储中心点坐标的列表
center_points = []

file_path= 'G:\\lidarDATA\\zone50\\demo\\'   # 输出文件路径------------------------------------------------------------------------>gaidongchu
file_name=get_all_filenames(file_path)
'''
for name in file_name:
    print_axis(file_path + name + ".las")
    coords = print_axis(file_path+name+".las")
    print("start",name)
    if "_转换为Las" in name:
        modified_name = name.replace("_转换为Las", "")
    else:
        modified_name = name
    # 将中心点坐标添加到列表中
    center_points.append((modified_name, coords[0],coords[1]))

# 创建数据框
df = pd.DataFrame(center_points, columns=['name', 'center_lon', 'center_lat'])

#%% 保存数据框为 CSV 文件
output_file = "G:\lidarDATA\ZONE50.csv"  # 输出文件路径---------------------------------------------------------------------->gaidongchu
df.to_csv(output_file, index=False)
'''
#%%
output_path= r"G:\lidarDATA\CBH_ZONE50"
for name in file_name:
    print(name)
    las_CBH_create(file_path + name + ".las",output_path + name + ".csv")

