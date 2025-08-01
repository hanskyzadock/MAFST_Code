from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import numpy as np
from config.paths import PathConfig
from config.params import ModelParams


class ModelTrainer:
    def __init__(self):
        self.pipeline = self._build_pipeline()
        self.feature_names = None

    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectFromModel(
                XGBClassifier(**ModelParams.XGB_PARAMS),
                threshold=ModelParams.FEATURE_SELECT_THRESH
            )),
            ('classifier', XGBClassifier(**ModelParams.XGB_PARAMS))
        ])

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """训练模型并返回评估指标"""
        self._init_feature_names(X.shape[1])
        cv_metrics = self._cross_validate(X, y)
        self._save_model()
        return cv_metrics

    def _init_feature_names(self, n_raw_features: int):
        """初始化特征名称（用于可解释性分析）"""
        # 菌丝特征
        self.feature_names = [
            'cell_state_germination_ratio', 'cell_state_germinated_mean',
            'cell_state_overall_mean', 'cell_state_germinated_std'
        ]
        # 其他维度特征
        for dim in ['mil', 'cell_size']:
            self.feature_names.extend([
                f"{dim}_mean", f"{dim}_std",
                f"{dim}_median", f"{dim}_iqr"
            ])

    def _cross_validate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """执行交叉验证"""
        cv = StratifiedKFold(ModelParams.N_FOLDS, shuffle=True, random_state=ModelParams.RANDOM_STATE)
        metrics = {'auc': [], 'accuracy': []}

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            self.pipeline.fit(X_train, y_train)
            probs = self.pipeline.predict_proba(X_test)[:, 1]

            metrics['auc'].append(roc_auc_score(y_test, probs))
            metrics['accuracy'].append(accuracy_score(y_test, probs > 0.5))

        return metrics

    def export_formula(self, decimal: int = 3) -> str:
        """
        生成完整的数学公式表达式
        格式示例：
        Z = 0.352*(Cell_state_germination_ratio-0.42)/0.18
           + 0.201*(MIL_mean-15.3)/2.1
           + ...
        """
        scaler = self.pipeline.named_steps['scaler']
        selector = self.pipeline.named_steps['selector']
        classifier = self.pipeline.named_steps['classifier']

        # 获取被选中的特征索引
        selected_idx = selector.get_support(indices=True)

        formula_terms = []
        for idx, imp in zip(selected_idx, classifier.feature_importances_):
            # 获取标准化参数
            mean = scaler.mean_[idx]
            std = scaler.scale_[idx]
            feat_name = self.feature_names[idx]

            term = f"{imp:.{decimal}f} * (({feat_name} - {mean:.2f}) / {std:.2f})"
            formula_terms.append(term)

        formula = "综合指数 Z = \n" + " +\n".join(formula_terms)
        return formula

    def _save_model(self):
        """保存模型到指定路径"""
        joblib.dump(self.pipeline, PathConfig.MODEL_DIR / "best_model4.pkl")