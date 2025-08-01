import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import seaborn as sns


class ShapAnalyzer:
    def __init__(
            self,
            pipeline,
            original_feature_names: List[str],
            plot_params: Optional[Dict] = None
    ):
        self.pipeline = pipeline
        self.original_feature_names = original_feature_names
        self.selected_feature_names = self._get_selected_features()
        self.plot_params = plot_params or {}

        # 设置默认绘图参数
        self._default_params = {
            'summary_cmap': 'bwr',
            'dependence_cmap': 'bwr',
            'bar_palette': 'viridis',
            'font_size': 16,
            'label_color': '#444444',
            'title_fontsize': 18,
            'title_color': '#444444',
            'figure_dpi': 300
        }

    def _get_plot_param(self, param_name: str):
        """获取绘图参数，优先使用用户自定义值"""
        return self.plot_params.get(param_name, self._default_params[param_name])

    def _get_selected_features(self) -> List[str]:
        """获取被选中的特征名称"""
        selector = self.pipeline.named_steps['selector']
        mask = selector.get_support()
        return [name for name, keep in zip(self.original_feature_names, mask) if keep]

    def analyze(self, X_raw: np.ndarray, data_source: str, save_dir: Path) -> pd.DataFrame:
        """执行SHAP分析"""
        # 数据预处理
        X_transformed = self.pipeline.named_steps['scaler'].transform(X_raw)
        X_transformed = self.pipeline.named_steps['selector'].transform(X_transformed)

        # 初始化解释器
        model = self.pipeline.named_steps['classifier']
        explainer = shap.TreeExplainer(model)

        # 计算SHAP值
        shap_values = explainer.shap_values(X_transformed)

        # 生成汇总统计
        summary_df = self._create_summary_df(shap_values, data_source)

        # 可视化输出
        self._plot_summary(shap_values, X_transformed, save_dir)
        self._plot_importance_bar(summary_df, save_dir)
        self._plot_dependence(shap_values, X_transformed, save_dir)

        return summary_df

    def _create_summary_df(self, shap_values: np.ndarray, data_source: str) -> pd.DataFrame:
        """生成SHAP汇总统计表"""
        summary_data = []
        for i, feat_name in enumerate(self.selected_feature_names):
            summary_data.append({
                'feature': feat_name,
                'mean_abs_shap': np.mean(np.abs(shap_values[:, i])),
                'mean_shap': np.mean(shap_values[:, i]),
                'data_source': data_source
            })
        return pd.DataFrame(summary_data).sort_values('mean_abs_shap', ascending=False)

    def _apply_style_settings(self):
        """应用全局样式设置"""
        plt.rc('font', size=self._get_plot_param('font_size'))
        plt.rc('axes', labelsize=self._get_plot_param('font_size'),
               labelcolor=self._get_plot_param('label_color'))
        plt.rc('axes', titlesize=self._get_plot_param('title_fontsize'),
               titlecolor=self._get_plot_param('title_color'))

    def _plot_summary(self, shap_values: np.ndarray, X: np.ndarray, save_dir: Path):
        """绘制SHAP摘要图"""
        self._apply_style_settings()
        plt.figure(figsize=(12, 6), dpi=self._get_plot_param('figure_dpi'))
        shap.summary_plot(
            shap_values, X,
            feature_names=self.selected_feature_names,
            plot_type="dot",
            show=False,
            color_bar=False,  # 禁用自动颜色条以自定义样式
            #cmap=self._get_plot_param('summary_cmap')
        )

        # 手动添加颜色条并设置样式
        cb = plt.gcf().axes[-1]
        cb.set_ylabel('Feature value',
                      fontsize=self._get_plot_param('font_size'),
                      color=self._get_plot_param('label_color'))

        plt.savefig(save_dir / "SHAP_Summary.png", bbox_inches='tight')
        plt.close()

    def _plot_dependence(self, shap_values: np.ndarray, X: np.ndarray, save_dir: Path):
        """绘制前10特征依赖图"""
        self._apply_style_settings()
        top_features = np.argsort(np.mean(np.abs(shap_values), axis=0))[::-1][:10]

        for idx in top_features:
            plt.figure(figsize=(8, 5), dpi=self._get_plot_param('figure_dpi'))
            shap.dependence_plot(
                idx, shap_values, X,
                feature_names=self.selected_feature_names,
                show=False,
                #cmap=self._get_plot_param('dependence_cmap')
            )

            # 设置坐标轴样式
            ax = plt.gca()
            ax.set_xlabel(ax.get_xlabel(),
                          fontsize=self._get_plot_param('font_size'),
                          color=self._get_plot_param('label_color'))
            ax.set_ylabel(ax.get_ylabel(),
                          fontsize=self._get_plot_param('font_size'),
                          color=self._get_plot_param('label_color'))

            plt.savefig(save_dir / f"SHAP_Dependence_{self.selected_feature_names[idx]}.png",
                        bbox_inches='tight')
            plt.close()

    def _plot_importance_bar(self, summary_df: pd.DataFrame, save_dir: Path):
        """绘制特征重要性条形图"""
        self._apply_style_settings()
        plt.figure(figsize=(10, 6), dpi=self._get_plot_param('figure_dpi'))
        summary_df = summary_df.sort_values('mean_abs_shap', ascending=False)

        ax = sns.barplot(
            x='mean_abs_shap',
            y='feature',
            data=summary_df.head(15),
            palette=self._get_plot_param('bar_palette')
        )

        # 设置坐标轴和标题
        ax.set_xlabel('Mean |SHAP Value|',
                      fontsize=self._get_plot_param('font_size'),
                      color=self._get_plot_param('label_color'))
        ax.set_ylabel('Feature',
                      fontsize=self._get_plot_param('font_size'),
                      color=self._get_plot_param('label_color'))
        ax.set_title('SHAP Feature Importance',
                     fontsize=self._get_plot_param('title_fontsize'),
                     color=self._get_plot_param('title_color'))

        plt.tight_layout()
        plt.savefig(save_dir / "SHAP_Importance_Rank.png", bbox_inches='tight')
        plt.close()