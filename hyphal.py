import numpy as np


class HyphalFeatureEngineer:
    @staticmethod
    def calculate(dim_data: np.ndarray) -> dict:
        """
        计算菌丝特征：
        1. 萌发率：菌丝长度>0的细胞比例
        2. 萌发细胞平均长度（仅统计>0的细胞）
        3. 全体细胞平均长度（包括未萌发细胞）
        4. 萌发长度标准差（仅统计>0的细胞）
        """
        # 转换为浮点型确保计算安全
        dim_data = dim_data.astype(float)

        # 萌发细胞判断
        germinated = dim_data > 0
        germinated_data = dim_data[germinated]
        n_germinated = len(germinated_data)

        # 萌发率
        germination_ratio = n_germinated / len(dim_data) if len(dim_data) > 0 else 0.0

        # 萌发细胞统计量
        if n_germinated == 0:
            germinated_mean = 0.0
            germinated_std = 0.0
        else:
            germinated_mean = float(np.mean(germinated_data))
            germinated_std = float(np.std(germinated_data))

        # 全体细胞统计量
        overall_mean = float(np.mean(dim_data))

        return {
            "hyphal_germination_ratio": germination_ratio,
            "hyphal_germinated_mean": germinated_mean,
            "hyphal_overall_mean": overall_mean,
            "hyphal_germinated_std": germinated_std
        }