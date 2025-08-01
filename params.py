# 模型参数
class ModelParams:
    # XGBoost参数
    XGB_PARAMS = {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }

    # 特征选择阈值
    FEATURE_SELECT_THRESH = "median"

    # 交叉验证设置
    N_FOLDS = 5
    RANDOM_STATE = 42