import pandas as pd


def validate_csv_structure(df: pd.DataFrame) -> None:
    """严格验证CSV文件结构"""
    required_columns = {'MIL', 'Cell_size', 'Cell_state'}
    missing_cols = required_columns - set(df.columns)
    extra_cols = set(df.columns) - required_columns

    if missing_cols:
        raise ValueError(f"CSV缺少必要列: {missing_cols}")
    if extra_cols:
        raise ValueError(f"CSV包含非法列: {extra_cols}")

    for col in df.columns:
        valid_count = df[col].dropna().shape[0]
        # if valid_count < 19:
        #     raise ValueError(f"列 {col} 有效数据不足20个 (当前: {valid_count})")
        if df[col].max() > 1e4:
            raise ValueError(f"列 {col} 存在超过10000的异常值")