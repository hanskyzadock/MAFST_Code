import numpy as np
from scipy.stats import skew, kurtosis

class BaseFeatureEngineer:
    @staticmethod
    def calculate(dim_data: np.ndarray) -> dict:
        """计算常规维度特征"""
        return {
            "mean": float(np.mean(dim_data)),
            "std": float(np.std(dim_data)),
            "median": float(np.median(dim_data)),
            "IQR": float(np.subtract(*np.percentile(dim_data, [75, 25]))),
            "skewness": float(skew(dim_data)),
            "kurtosis": float(kurtosis(dim_data))
        }