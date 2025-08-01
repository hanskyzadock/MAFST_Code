# coding = UTF-8
import pandas as pd
from scipy.stats import pearsonr
import numpy as np


def pearsonr_correlation(file_path, output_file):
    # 读取数据
    specs = pd.read_csv(file_path, sep='\t', header=None)

    # 初始化存储数组
    correlations = []
    p_values = []

    # 计算相关系数和p值
    for index in specs.index[1:]:
        correlation, p_value = pearsonr(specs.loc[index, 1:], specs.loc[0, 1:])
        correlations.append(correlation)
        p_values.append(p_value)

    # 构建结果矩阵
    row, col = specs.shape
    pearsonr_cor = np.empty((row, 3), dtype=object)
    pearsonr_cor[0, :] = ['Wavenumbers', 'Pearsonr_correlation', 'p_values']

    # 填充数据并根据p值过滤
    for i in range(1, row):
        pearsonr_cor[i, 0] = specs.loc[i, 0]  # 始终保留波数

        if p_values[i - 1] < 0.05:  # 注意索引偏移
            pearsonr_cor[i, 1] = correlations[i - 1]
            pearsonr_cor[i, 2] = p_values[i - 1]
        else:
            pearsonr_cor[i, 1] = ""  # 清除非显著值
            pearsonr_cor[i, 2] = ""

    # 保存结果（使用特定格式处理空值）
    np.savetxt(output_file, pearsonr_cor,
               delimiter='\t',
               fmt='%s\t%s\t%s',  # 保持三列格式
               header='',
               comments='')


# 使用示例
file_path = r'G:\python_test\Candida\correction\MCF_finger_zscore_368&363.txt'
output_file = r'G:\python_test\Candida\correction\MCF_finger_zscore_368&363_cor.csv'
pearsonr_correlation(file_path, output_file)