import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


class DataSummarizer:
    @staticmethod
    def summarize_dataset(samples: Dict, pred_df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        summary_data = []
        for sample_id, data in samples.items():
            pred_match = pred_df[pred_df['filename'] == sample_id]
            if pred_match.empty:
                print(f"警告: 样本 {sample_id} 在预测结果中未找到，已跳过")
                continue

            pred_info = pred_match.iloc[0]
            stats = {
                'filename': sample_id,
                'dataset': dataset_type,
                'MIL_mean': np.mean(data['data']['MIL']),
                'MIL_std': np.std(data['data']['MIL']),
                'MIL_median': np.median(data['data']['MIL']),
                'Cell_size_mean': np.mean(data['data']['Cell_size']),
                'Cell_size_std': np.std(data['data']['Cell_size']),
                'Cell_size_median': np.median(data['data']['Cell_size']),
                'Cell_state_mean': np.mean(data['data']['Cell_state']),
                'Cell_state_std': np.std(data['data']['Cell_state']),
                'pred_score': pred_info['score'],  # 修改列名匹配
                'pred_class': "R" if pred_info['pred_label'] == 1 else "S"  # 修改列名匹配
            }
            summary_data.append(stats)
        return pd.DataFrame(summary_data)