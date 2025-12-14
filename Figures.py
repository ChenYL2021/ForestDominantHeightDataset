# -*- coding: utf-8 -*-
# @Time    : 2024/4/29 10:46
# @Author  : ChenYuling
# @FileName: figures.py
# @Software: PyCharm
# @Describe：密度散点图
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rcParams
from statistics import mean
from sklearn.metrics import explained_variance_score,r2_score,median_absolute_error,mean_squared_error,mean_absolute_error
from scipy.stats import pearsonr
config = {"font.family":'Times New Roman',"font.size": 20,"mathtext.fontset":'stix'}
import pandas as pd

#%%
# df = pd.read_csv('./DATA/H_HEBING.csv',encoding="gb2312")#,encoding="gb2312"
# # df1= df.query("pnf==1")
# x = df['Hw']; y = df['H3pre']

#%%
df = pd.read_csv('./DATA/HTdat.csv')#,encoding="gb2312"
x = df['HT']; y = df['predict_HT']
# #
#%% 计算散点密度
xy = np.vstack([x, y])
z = stats.gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x.iloc[idx], y.iloc[idx], z[idx]
#%% 保存为.npy文件
np.save(r'E:\TH\FIGS\FIG3\DATA\z_HT.npy', z)
np.save(r'E:\TH\FIGS\FIG3\DATA\y_HT.npy', y)
np.save(r'E:\TH\FIGS\FIG3\DATA\x_HT.npy', x)

#%%
df = pd.read_csv('./DATA/HTdat.csv',encoding="gb2312")#
x1 = df['HT']; y1 = df['predict_HT']
# #
#%% 计算散点密度
xy1 = np.vstack([x1, y1])
z1 = stats.gaussian_kde(xy1)(xy1)
idx = z1.argsort()
x1, y1, z1 = x1.iloc[idx], y1.iloc[idx], z1[idx]
#%% 保存为.npy文件
np.save(r'G:\TH\FIGS\FIG3\DATA\z_H1.npy', z1)
np.save(r'G:\TH\FIGS\FIG3\DATA\y_H1.npy', y1)
np.save(r'G:\TH\FIGS\FIG3\DATA\x_H1.npy', x1)

#%%LOAD
z = np.load(r'E:\TH\FIGS\FIG3\DATA\z_HT.npy')
y = np.load(r'E:\TH\FIGS\FIG3\DATA\y_HT.npy')
x = np.load(r'E:\TH\FIGS\FIG3\DATA\x_HT.npy')
#%%LOAD
z1 = np.load(r'G:\TH\FIGS\FIG3\DATA\z_H3.npy')
y1 = np.load(r'G:\TH\FIGS\FIG3\DATA\y_H3.npy')
x1 = np.load(r'G:\TH\FIGS\FIG3\DATA\x_H3.npy')

#%%
# 拟合（若换MK，自行操作）最小二乘
def slope(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b

k, b = slope(x, y)
regression_line = []
for a in x:
    regression_line.append((k * a) + b)
#%%
k1, b1 = slope(x1, y1)
regression_line1 = []
for a1 in x1:
    regression_line1.append((k1 * a1) + b1)
#%%
BIAS = mean(x - y)
MSE = mean_squared_error(x, y)
RMSE = np.power(MSE, 0.5)
r = pearsonr(x, y).statistic
MAE = mean_absolute_error(x, y)
print(RMSE,r)
#%%

BIAS1 = mean(x1 - y1)
MSE1 = mean_squared_error(x1, y1)
RMSE1 = np.power(MSE1, 0.5)
r1 = pearsonr(x1, y1).statistic
MAE1 = mean_absolute_error(x1, y1)
# BIAS: 偏差的均值
# MSE: 均方误差
# RMSE: 均方根误差
# R2: Pearson相关系数
# adjR2: 调整后的R²
# MAE: 平均绝对误差
# EV: 解释的方差分数
# NSE: Nash-Sutcliffe效率
print(RMSE,r)
#%%
fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
# plt.plot(x, lower_confidence_bound, linestyle='--', color='black', dashes=(1, 40), label='95% Prediction Band')
# plt.plot(x, upper_confidence_bound, linestyle='--', color='black', dashes=(1, 40)), cmap='RdBu_r','Spectral'
scatter = ax.scatter(x, y, marker='o', c=z * 100, edgecolors=None, s=10,   alpha=0.8)
# 添加颜色条并设置字体大小
cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30)
cbar.set_label('Frequency', fontsize=16)  # 设置颜色条标签字体大小
cbar.ax.tick_params(labelsize=16)  # 设置颜色条刻度值字体大小
plt.plot([0, 80], [0, 80], 'red', lw=1.5, linestyle='--', label='1:1 line')#, label='1:1 line'
plt.plot(x, regression_line, 'black', lw=1.5, label='Regression Line')
ax.grid(True, linestyle='--', alpha=0.2)

plt.text(3,75, '$r = %.3f$' % r, family = 'Times New Roman', horizontalalignment='left', fontsize=20)
# plt.text(3,62, '$NSE=%.3f$' % NSE, family = 'Times New Roman', horizontalalignment='left')
plt.text(3,70, '$MAE = %.3f$' % MAE, family = 'Times New Roman', horizontalalignment='left', fontsize=20)
plt.text(3,65, '$RMSE = %.3f$' % RMSE, family = 'Times New Roman', horizontalalignment='left', fontsize=20)

plt.axis([0, 80, 0, 80])  # 设置线的范围
plt.xlabel('h$_d$ (m)', fontsize=20)  # 改变坐标轴标题字体
plt.ylabel('preh$_d$ (m)', fontsize=20)  # 改变坐标轴标题字体v
ax.legend(loc='lower right', frameon = False, fontsize=16)
# 设置刻度值数字的字体大小
ax.tick_params(axis='both', which='major', labelsize=16)
plt.savefig(r"E:\TH\FIGS\FIG3\HTnew.jpg", dpi=600, format='jpeg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中

# plt.savefig(r"G:\TH\FIGS\FIG3\H3\H3.svg", dpi=600, format='svg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()

#%%
plt.savefig(r"E:\TH\FIGS\FIG3\HTnew.jpg", dpi=600, format='jpeg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中

#%%
fig, ax = plt.subplots(figsize=(8, 6), dpi=600)

# 绘制散点图和其他元素
scatter = ax.scatter(x, y, marker='o', c=z * 100, edgecolors=None, s=10, alpha=0.8)
# 添加颜色条并设置字体大小
cbar = plt.colorbar(scatter, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30)
cbar.set_label('Frequency', fontsize=16)  # 设置颜色条标签字体大小
plt.plot([0, 80], [0, 80], 'red', lw=1.5, linestyle='--', label='1:1 line')
plt.plot(x, regression_line, 'black', lw=1.5, label='Regression Line')
ax.grid(True, linestyle='--', alpha=0.2)

# 添加文本
plt.text(3, 75, '$r = %.3f$' % r, family='Times New Roman', horizontalalignment='left')
plt.text(3, 72, '$MAE = %.3f$' % MAE, family='Times New Roman', horizontalalignment='left')
plt.text(3, 69, '$RMSE = %.3f$' % RMSE, family='Times New Roman', horizontalalignment='left')

# 设置轴的范围和标签
plt.axis([0, 80, 0, 80])
plt.xlabel('h$_d$ (m)')
plt.ylabel('preh$_d$ (m)')
ax.legend(loc='lower right', frameon=False)

# 保存图形为PDF格式，去掉外边界白色区域
plt.savefig(r"E:\TH\FIGS\FIG3\HT.pdf", dpi=600, format='pdf', pad_inches=0.02, bbox_inches='tight')

plt.show()

#%%
plt.savefig(r"G:\TH\FIGS\FIG3\H3\H3.pdf", dpi=600, format='pdf',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()


#%%
import os

# 检查路径是否存在
output_dir = r"G:\TH\FIGS\FIG3\H3"
if not os.path.exists(output_dir):
    print(f"Directory {output_dir} does not exist.")
else:
    # 尝试保存文件
    output_file = os.path.join(output_dir, "H3.pdf")
    plt.savefig(output_file, dpi=600, format='pdf', pad_inches=0.02, bbox_inches='tight')
    plt.close()
    print(f"File saved successfully to {output_file}")



#%%
from matplotlib.gridspec import GridSpec
# 创建一个包含两个子图的图形
fig = plt.figure(figsize=(16, 6))  # 设置整体图形的大小
# 使用GridSpec定义子图布局
gs = GridSpec(1, 2, width_ratios=[1, 1])  # 第一个子图是第二个子图宽度的两倍
# # 创建一个包含两个子图的图形
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
# 字体大小设置
label_fontsize = 18
title_fontsize = 16
tick_fontsize = 16
# num_points=203447
# # 设置 colorbar 的刻度值和标签
# num_ticks = 5  # 想要的刻度数量
# tick_positions1 = np.linspace(0, 2, num_ticks)  # 在0到1之间等间隔取刻度位置
# tick_labels1 = [f'{int(pos * num_points)}' for pos in tick_positions1]  # 刻度标签为对应点数量
#
# tick_positions2 = np.linspace(0, 1.7, num_ticks)  # 在0到1之间等间隔取刻度位置
# tick_labels2 = [f'{int(pos * num_points)}' for pos in tick_positions2]  # 刻度标签为对应点数量
# 在第一个子图中绘制数据
ax1 = fig.add_subplot(gs[0])
scatter = ax1.scatter(x, y, marker='o', c=z * 100, edgecolors=None, s=10, alpha=0.8)
cbar1 = fig.colorbar(scatter, ax=ax1, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30)
cbar1.set_ticks([])  # 取消刻度值
# cbar1.set_ticks(tick_positions1)  # 设置刻度值
# cbar1.set_ticklabels(tick_labels1)  # 设置刻度标签

ax1.plot([0, 70], [0, 70], 'red', lw=1.5, linestyle='--', label='1:1 line')  # 1:1 line
ax1.plot(x, regression_line, 'black', lw=1.5, label='Regression Line')
ax1.grid(True, linestyle='--', alpha=0.2)
ax1.text(3, 65, 'r = %.3f' % r, family='Times New Roman', fontsize=label_fontsize, horizontalalignment='left')
ax1.text(3, 60, 'MAE = %.1f' % MAE, family='Times New Roman', fontsize=label_fontsize, horizontalalignment='left')
ax1.text(3, 55, 'RMSE = %.1f' % RMSE, family='Times New Roman', fontsize=label_fontsize, horizontalalignment='left')
ax1.set_xlim(0, 70)
ax1.set_ylim(0, 70)
ax1.set_xlabel(r'$h_{a}$ (m)', fontsize=label_fontsize)  # 改变坐标轴标题字体
ax1.set_ylabel('Predicted $h_{a}$ (m)', fontsize=label_fontsize)  # 改变坐标轴标题字体
ax1.legend(loc='lower right', frameon=False, fontsize=tick_fontsize)
ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)

# 在第二个子图中绘制数据
ax2 = fig.add_subplot(gs[1])
scatter1 = ax2.scatter(x1, y1, marker='o', c=z1 * 100, edgecolors=None, s=10, alpha=0.8)
cbar2 = fig.colorbar(scatter1, ax=ax2, shrink=1, orientation='vertical', extend='both', pad=0.015, aspect=30)
cbar2.set_ticks([])  # 取消刻度值
# cbar2.set_ticks(tick_positions2)  # 设置刻度值
# cbar2.set_ticklabels(tick_labels2)  # 设置刻度标签
ax2.plot([0, 70], [0, 70], 'red', lw=1.5, linestyle='--', label='1:1 line')  # 1:1 line
ax2.plot(x1, regression_line1, 'black', lw=1.5, label='Regression Line')
ax2.grid(True, linestyle='--', alpha=0.2)
ax2.text(3, 65, 'r = %.3f' % r1, family='Times New Roman', fontsize=label_fontsize, horizontalalignment='left')
ax2.text(3, 60, 'MAE = %.1f' % MAE1, family='Times New Roman', fontsize=label_fontsize, horizontalalignment='left')
ax2.text(3, 55, 'RMSE = %.1f' % RMSE1, family='Times New Roman', fontsize=label_fontsize, horizontalalignment='left')
ax2.set_xlim(0, 70)
ax2.set_ylim(0, 70)
ax2.set_xlabel('$h_{w}$ (m)', fontsize=label_fontsize)  # 改变坐标轴标题字体
ax2.set_ylabel('Predicted $h_{w}$ (m)', fontsize=label_fontsize)  # 改变坐标轴标题字体
ax2.legend(loc='lower right', frameon=False, fontsize=tick_fontsize)
ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)

# 调整布局以防止重叠
plt.tight_layout()

#%% 保存图形为PDF文件
output_file = r"G:\TH\FIGS\MSfigs\Fig5.pdf"
plt.savefig(output_file, dpi=600, format='pdf', pad_inches=0.02, bbox_inches='tight')
plt.savefig(r"G:\TH\FIGS\MSfigs\Fig5.jpg", dpi=600, format='jpeg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中

# 显示图形
plt.show()

# 关闭图形
plt.close()

print(f"File saved successfully to {output_file}")


#%%
import os

# 检查路径是否存在
output_dir = r"G:\TH\FIGS\FIG3"
if not os.path.exists(output_dir):
    print(f"Directory {output_dir} does not exist.")
else:
    # 尝试保存文件
    output_file = os.path.join(output_dir, "H.pdf")
    plt.savefig(output_file, dpi=600, format='pdf', pad_inches=0.02, bbox_inches='tight')
    plt.close()
    print(f"File saved successfully to {output_file}")




#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
# 计算拟合线
slope, intercept, r_value, p_value, std_err = linregress(x, y)
regression_line = slope * x + intercept
# 计算置信区间
confidence_interval = 1.96 * std_err * np.sqrt(1/len(x) + (x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
lower_confidence_bound = regression_line - confidence_interval
upper_confidence_bound = regression_line + confidence_interval

fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
# plt.plot(x, lower_confidence_bound, linestyle='--', color='black', dashes=(1, 40), label='95% Prediction Band')
# plt.plot(x, upper_confidence_bound, linestyle='--', color='black', dashes=(1, 40)), cmap='RdBu_r','Spectral'
scatter = ax.scatter(x, y, marker='o',facecolors='#F7776C', edgecolors='#F7776C', s=12, linewidth=1, alpha=0.5)# c=z * 100,
# 添加置信区间
ax.fill_between(x, lower_confidence_bound, upper_confidence_bound, color='#F7776C', alpha=0.2, label='95% Confidence Interval')

plt.plot([0, 30], [0, 30], "#F7776C", lw=0.5, linestyle='--', label='1:1 line')#, label='1:1 line'
plt.plot(x, regression_line, 'black', lw=0.8, label='Regression Line')
ax.grid(True, linestyle='--', alpha=0.2)

plt.text(1.5,28, '$r=%.3f$' % r, family = 'Times New Roman', horizontalalignment='left')
# plt.text(3,62, '$NSE=%.3f$' % NSE, family = 'Times New Roman', horizontalalignment='left')
plt.text(1.5,26, '$MAE =%.1f$' % MAE, family = 'Times New Roman', horizontalalignment='left')
plt.text(1.5,24, '$RMSE=%.1f$' % RMSE, family = 'Times New Roman', horizontalalignment='left')

plt.axis([0, 30, 0, 30])  # 设置线的范围
plt.xlabel('h$_w$ (m)')  # 改变坐标轴标题字体
plt.ylabel('preh$_w$ (m)')  # 改变坐标轴标题字体
ax.legend(loc='lower right', frameon = False)
# plt.savefig(r"G:\TH\FIGS\FIG3\H3\weight.svg", dpi=600, format='svg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.show()
#%%
plt.savefig(r"G:\TH\FIGS\FIG3\weight.jpg", dpi=600, format='jpeg',pad_inches=0.02,bbox_inches='tight')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.close()



#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# 读取数据
df = pd.read_csv('./DATA/H_HEBING.csv', encoding="gb2312")
x = df['Hw']
y = df['H3pre']
import seaborn as sns
from matplotlib import pyplot as plt
tips = sns.load_dataset("tips")
plt.figure(figsize=(5, 5))
sns.lmplot(x="Hw",y="H3pre",data=df)
plt.savefig(u'D://test2.pdf')
#%%
# 排序数据以确保 fill_between 正常工作
sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_sorted = y[sorted_indices]

# 计算拟合线
slope, intercept, r_value, p_value, std_err = linregress(x_sorted, y_sorted)
regression_line = slope * x_sorted + intercept

# 计算置信区间
confidence_interval = 1.96 * std_err * np.sqrt(1/len(x_sorted) + (x_sorted - np.mean(x_sorted))**2 / np.sum((x_sorted - np.mean(x_sorted))**2))
lower_confidence_bound = regression_line - confidence_interval
upper_confidence_bound = regression_line + confidence_interval

# 绘制散点图和拟合线
fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
scatter = ax.scatter(x_sorted, y_sorted, marker='o', facecolors='white', edgecolors='black', s=12, linewidth=1, alpha=0.5)

# 添加置信区间
ax.fill_between(x_sorted, lower_confidence_bound, upper_confidence_bound, color='#F7776C', alpha=0.8, label='95% Confidence Interval')

# 添加拟合线和1:1线
plt.plot([0, 30], [0, 30], "#F7776C", lw=0.5, linestyle='--', label='1:1 line')
plt.plot(x_sorted, regression_line, 'black', lw=0.8, label='Regression Line')
ax.grid(True, linestyle='--', alpha=0.2)

# 计算 MAE 和 RMSE
mae = np.mean(np.abs(y_sorted - regression_line))
rmse = np.sqrt(np.mean((y_sorted - regression_line)**2))

# 添加文本信息
plt.text(1.5, 28, '$r=%.3f$' % r_value, family='Times New Roman', horizontalalignment='left')
plt.text(1.5, 26, '$MAE=%.1f$' % mae, family='Times New Roman', horizontalalignment='left')
plt.text(1.5, 24, '$RMSE=%.1f$' % rmse, family='Times New Roman', horizontalalignment='left')

# 设置坐标轴范围和标签
plt.axis([0, 30, 0, 30])
plt.xlabel('h$_w$ (m)', family='Times New Roman')
plt.ylabel('preh$_w$ (m)', family='Times New Roman')
ax.legend(loc='lower right', frameon=False)

# 显示图表
plt.show()

#%%
import PyPDF2
import fitz  # PyMuPDF
#%%
import PyPDF2
import fitz  # PyMuPDF

# 合并两个 PDF 文件
def merge_pdfs(pdf_list, output):
    pdf_writer = PyPDF2.PdfWriter()

    for pdf in pdf_list:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_writer.add_page(page)

    with open(output, 'wb') as out:
        pdf_writer.write(out)

# 去除 PDF 页面空白区域
def remove_whitespace(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        rect = page.rect
        pix = page.get_pixmap()
        img = fitz.Pixmap(pix, 0) if pix.alpha else pix
        new_rect = img.irect
        page.set_cropbox(fitz.Rect(new_rect))
    doc.save(output_path)

# 示例使用
pdf_list = [r'G:\TH\FIGS\FIG9\ha.pdf', r'G:\TH\FIGS\FIG9\hw.pdf']
merged_pdf = r'G:\TH\FIGS\FIG9\haw.pdf'
final_pdf = r'G:\TH\FIGS\FIG9\haw1.pdf'

# 合并 PDF 文件
merge_pdfs(pdf_list, merged_pdf)

# 去除空白背景
remove_whitespace(merged_pdf, final_pdf)



#%%
vdata = pd.read_csv(r'G:\TH\FIGS\FIG11\PLOT_H.csv', sep=',', encoding = 'gb2312')

a =vdata.describe()