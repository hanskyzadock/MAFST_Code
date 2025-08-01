from config.paths import PathConfig
from config.params import ModelParams
from src.data_loader import load_all_samples
from src.features.hyphal import HyphalFeatureEngineer
from src.features.base import BaseFeatureEngineer
from src.model.train import ModelTrainer
from src.model.evaluate import ModelVisualizer
import numpy as np
import pandas as pd
from src.analyze.shap_analyzer import ShapAnalyzer
from src.analyze.data_summarizer import DataSummarizer

def main():
    # 初始化环境
    PathConfig.setup_dirs()

    # 数据加载与特征生成
    print("数据加载中……")
    samples = load_all_samples()
    print("数据加载完成")
    X, y, sample_ids = generate_features(samples)  # 新增接收 sample_ids

    # 模型训练与评估
    trainer = ModelTrainer()
    metrics = trainer.train(X, y)

    # 输出特征重要性
    print("\n=== 特征重要性 ===")
    xgb = trainer.pipeline.named_steps['classifier']
    for idx in np.argsort(xgb.feature_importances_)[::-1]:
        name = trainer.feature_names[idx]
        print(f"{name}: {xgb.feature_importances_[idx]:.3f}")

        # ==================== 可视化输出 ====================
    if PathConfig.REPORT_DIR:
        report_dir = PathConfig.REPORT_DIR / "analysis_results"
        report_dir.mkdir(exist_ok=True)

        # 自定义样式
        plot_style = {
            'fontsize': 12,
            'title_fontsize': 14,
            'color_map': 'viridis',
            'linewidth': 2
        }

        # 生成各子图
        ModelVisualizer.plot_roc(
            y_true=y,
            y_probs=trainer.pipeline.predict_proba(X)[:, 1],
            save_path=report_dir / "ROC_Curve.png",
             style=plot_style
        )

        ModelVisualizer.plot_feature_importance(
            feature_names=trainer.feature_names,
            importances=trainer.pipeline.named_steps['classifier'].feature_importances_,
            save_path=report_dir / "Feature_Importance.png",
            style=plot_style
        )

        ModelVisualizer.plot_score_distribution(
            y_true=y,
            y_probs=trainer.pipeline.predict_proba(X)[:, 1],
            save_path=report_dir / "Score_Distribution.png",
            style={'bins': 20, 'fontsize': 12}
        )

        ModelVisualizer.plot_correlation_heatmap(
            features=X,
            feature_names=trainer.feature_names,
            save_path=report_dir / "Correlation_Heatmap.png",
            style={'fontsize': 8, 'color_map': 'coolwarm'}
        )

        # 生成训练集预测结果
        y_train_pred = trainer.pipeline.predict(X)

        # 绘制训练集混淆矩阵
        train_cm_path = report_dir / "Train_Confusion_Matrix.png"
        train_acc = ModelVisualizer.plot_confusion_matrix(
            y_true=y,
            y_pred=y_train_pred,
            save_path=train_cm_path,
            style={'title': 'Train_Confusion_Matrix'}
        )
        print(f"\n训练集准确率: {train_acc:.2%}")

        # ==================== 公式输出 ====================
    formula_path = report_dir / "Sensitivity_Formula.txt"
    with open(formula_path, 'w', encoding='utf-8') as f:
        formula = trainer.export_formula(decimal=3)
        f.write("药敏综合指数计算公式：\n\n")
        f.write(formula)
        print(f"\n公式文件已保存至：{formula_path}")

    # 错误样本分析
    y_pred = trainer.pipeline.predict(X)
    wrong_indices = np.where(y_pred != y)[0]
    wrong_samples = [sample_ids[i] for i in wrong_indices]

    if wrong_samples:
        error_path = PathConfig.REPORT_DIR / "error_samples.txt"
        with open(error_path, 'w') as f:
            f.write("\n".join(wrong_samples))
        print(f"发现 {len(wrong_samples)} 个错误样本，已保存至 {error_path}")

    if PathConfig.REPORT_DIR:
        # 生成预测概率
        y_probs = trainer.pipeline.predict_proba(X)[:, 1]

        # 构建结果DataFrame
        train_result_df = pd.DataFrame({
            "filename": sample_ids,
            "true_label": y,
            "score": y_probs.round(4),
            "pred_label": (y_probs > 0.5).astype(int)
        })

        # 保存文件
        train_result_path = PathConfig.REPORT_DIR / "train_predictions4.csv"
        train_result_df.to_csv(train_result_path, index=False)
        print(f"\n训练集预测结果已保存至: {train_result_path}")

    if PathConfig.REPORT_DIR:
        shap_dir = PathConfig.REPORT_DIR / "shap_analysis"
        shap_dir.mkdir(exist_ok=True)

        # 初始化SHAP分析器
        analyzer = ShapAnalyzer(
            pipeline=trainer.pipeline,
            original_feature_names=trainer.feature_names
        )

        # 执行分析
        train_shap_df = analyzer.analyze(X, "train", shap_dir)

        # 保存汇总结果
        summary_path = shap_dir / "SHAP_Summary.csv"
        train_shap_df.to_csv(summary_path, index=False)
        print(f"\nSHAP分析结果已保存至: {summary_path}")

    if PathConfig.REPORT_DIR:
        # 训练集汇总
        train_summary_df = DataSummarizer.summarize_dataset(
            samples, train_result_df, "train"
        )

        # 保存结果
        summary_path = PathConfig.REPORT_DIR / "train_summary.csv"
        train_summary_df.to_csv(summary_path, index=False)
        print(f"训练集统计摘要已保存至: {summary_path}")

    if PathConfig.REPORT_DIR:
        # 假设测试集数据路径已知（根据项目实际情况调整）
        test_samples = load_all_samples()  # 若测试集需单独加载，需补充相应逻辑
        X_test, y_test, test_ids = generate_features(test_samples)
        X_full = np.concatenate([X, X_test])

        shap_dir = PathConfig.REPORT_DIR / "full_shap_analysis"
        shap_dir.mkdir(exist_ok=True)

        full_analyzer = ShapAnalyzer(
            pipeline=trainer.pipeline,
            original_feature_names=trainer.feature_names
        )
        full_shap_df = full_analyzer.analyze(X_full, "full_dataset", shap_dir)

        summary_path = shap_dir / "Full_SHAP_Summary.csv"
        full_shap_df.to_csv(summary_path, index=False)
        print(f"\n全数据集SHAP分析已保存至: {summary_path}")


def generate_features(samples: dict) -> tuple:
    features = []
    labels = []
    sample_ids = []  # 新增样本ID列表

    for sample_id, sample in samples.items():
        feat_vector = []

        # === 处理菌丝特征（N1维度） ===
        n1_data = sample['data']['Cell_state']
        hyphal_feats = HyphalFeatureEngineer.calculate(n1_data)
        feat_vector.extend([
            hyphal_feats["hyphal_germination_ratio"],
            hyphal_feats["hyphal_germinated_mean"],
            hyphal_feats["hyphal_overall_mean"],
            hyphal_feats["hyphal_germinated_std"]
        ])

        # === 处理其他维度 ===
        for dim in ['MIL', 'Cell_size']:
            dim_data = sample['data'][dim]
            base_feats = BaseFeatureEngineer.calculate(dim_data)
            feat_vector.extend([
                base_feats["mean"],
                base_feats["std"],
                base_feats["median"],
                base_feats["IQR"]
            ])

        features.append(feat_vector)
        labels.append(sample['label'])
        sample_ids.append(sample_id)  # 记录样本ID

    return np.array(features), np.array(labels), sample_ids

if __name__ == "__main__":
    main()