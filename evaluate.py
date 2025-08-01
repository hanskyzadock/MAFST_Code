from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict
from sklearn.metrics import roc_curve, auc,confusion_matrix, accuracy_score
import seaborn as sns
from typing import Optional


class ModelVisualizer:
    @staticmethod
    def plot_roc(
            y_true: np.ndarray,
            y_probs: np.ndarray,
            save_path: Path = None,
            figsize: tuple = (8, 6),
            dpi: int = 300,
            style: Dict = None
    ) -> float:
        """单独保存ROC曲线"""
        default_style = {
            'fontsize': 12,
            'linewidth': 2,
            'color': 'darkorange'
        }
        if style: default_style.update(style)

        plt.figure(figsize=figsize, dpi=dpi)
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr,
                 color=default_style['color'],
                 lw=default_style['linewidth'],
                 label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate', fontsize=default_style['fontsize'])
        plt.ylabel('True Positive Rate', fontsize=default_style['fontsize'])
        plt.title('ROC Curve', fontsize=default_style['fontsize'] + 2)
        plt.legend(loc="lower right")

        if save_path:
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"ROC曲线已保存至: {save_path}")
        plt.close()
        return roc_auc

    @staticmethod
    def plot_feature_importance(
            feature_names: list,
            importances: np.ndarray,
            save_path: Path = None,
            figsize: tuple = (10, 6),
            dpi: int = 300,
            style: Dict = None
    ):
        """单独保存特征重要性图"""
        default_style = {
            'color_map': 'viridis',
            'fontsize': 12,
            'title_fontsize': 14
        }
        if style:
            default_style.update(style)

        plt.figure(figsize=figsize, dpi=dpi)
        sorted_idx = np.argsort(importances)[::-1]

        # ========== 修复颜色映射部分 ==========
        cmap = plt.cm.get_cmap(default_style['color_map'])
        colors = cmap(np.linspace(0.3, 0.8, len(feature_names)))
        # ========== 修复结束 ==========

        plt.barh(np.array(feature_names)[sorted_idx],
                 importances[sorted_idx],
                 color=colors)
        plt.gca().invert_yaxis()
        plt.xlabel('Importance Score', fontsize=default_style['fontsize'])
        plt.title('Feature Importance', fontsize=default_style['title_fontsize'])

        if save_path:
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"特征重要性图已保存至: {save_path}")
        plt.close()

    @ staticmethod

    def plot_score_distribution(
            y_true: np.ndarray,
            y_probs: np.ndarray,
            save_path: Path = None,
            figsize: tuple = (10, 6),
            dpi: int = 300,
            style: Dict = None
    ):
        """单独保存评分分布图"""
        default_style = {
            'bins': 30,
            'kde': True,
            'fontsize': 12
        }
        if style: default_style.update(style)

        plt.figure(figsize=figsize, dpi=dpi)
        for label in [0, 1]:
            sns.histplot(
                y_probs[y_true == label],
                bins=default_style['bins'],
                kde=default_style['kde'],
                label='Resistant' if label else 'Sensitive'
            )
        plt.xlabel('Prediction Score', fontsize=default_style['fontsize'])
        plt.ylabel('Density', fontsize=default_style['fontsize'])
        plt.title('Score Distribution by Class', fontsize=default_style['fontsize'] + 2)
        plt.legend()

        if save_path:
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"评分分布图已保存至: {save_path}")
        plt.close()

    @staticmethod
    def plot_correlation_heatmap(
            features: np.ndarray,
            feature_names: list,
            save_path: Path = None,
            figsize: tuple = (12, 10),
            dpi: int = 300,
            style: Dict = None
    ):
        """单独保存相关性热力图"""
        default_style = {
            'annot': True,
            'fmt': ".2f",
            'color_map': 'coolwarm',
            'fontsize': 10
        }
        if style: default_style.update(style)

        plt.figure(figsize=figsize, dpi=dpi)
        corr_matrix = np.corrcoef(features.T)
        sns.heatmap(
            corr_matrix,
            annot=default_style['annot'],
            fmt=default_style['fmt'],
            cmap=default_style['color_map'],
            xticklabels=feature_names,
            yticklabels=feature_names
        )
        plt.title('Feature Correlation Heatmap', fontsize=default_style['fontsize'] + 2)

        if save_path:
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"相关性热力图已保存至: {save_path}")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(
            y_true: np.ndarray,
            y_pred: np.ndarray,
            save_path: Optional[Path] = None,
            figsize: tuple = (8, 6),
            dpi: int = 300,
            style: dict = None
    ) -> float:
        """可视化混淆矩阵并返回准确率"""
        default_style = {
            'fontsize': 12,
            'title': 'Confusion Matrix',
            'cmap': 'Blues',
            'labels': ['Sensitive(S)', 'Resistant(R)']
        }
        if style:
            default_style.update(style)

        plt.figure(figsize=figsize, dpi=dpi)
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap=default_style['cmap'],
            xticklabels=default_style['labels'],
            yticklabels=default_style['labels']
        )

        plt.xlabel('Prediction label', fontsize=default_style['fontsize'])
        plt.ylabel('True label', fontsize=default_style['fontsize'])
        plt.title(f"Prediction accuracy: {acc:.2%}",
                  fontsize=default_style['fontsize'] + 2)

        if save_path:
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"混淆矩阵已保存至: {save_path}")
        plt.close()

        return acc