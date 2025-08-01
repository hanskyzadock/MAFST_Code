from pathlib import Path
import pandas as pd
import argparse
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
from config.paths import PathConfig
from src.data_loader import load_sample_data
from src.features.hyphal import HyphalFeatureEngineer
from src.features.base import BaseFeatureEngineer
from src.model.evaluate import ModelVisualizer  # 新增评估模块导入
from src.analyze.data_summarizer import DataSummarizer



class DrugPredictor:
    def __init__(self, model_path: str):
        self.pipeline = joblib.load(model_path)

    def predict_single(self, csv_path: Path) -> float:
        """预测单个样本"""
        try:
            sample_data = load_sample_data(csv_path)
            features = self._generate_features(sample_data)
            return float(self.pipeline.predict_proba([features])[0, 1])
        except Exception as e:
            print(f"处理文件 {csv_path.name} 失败: {str(e)}")
            return np.nan

    def predict_batch(self, input_dir: Path, output_csv: Path, label_path: Path = None) -> None:
        """批量预测并保存结果（含标签对比和数据汇总）"""
        results = []
        y_true, y_pred = [], []

        # 加载标签（如果存在）
        label_map = {}
        if label_path and label_path.exists():
            label_df = pd.read_csv(label_path)
            label_map = label_df.set_index('filename')['label'].astype(int).to_dict()

        # 遍历目录下所有CSV文件
        for csv_file in input_dir.glob("*.csv"):
            score = self.predict_single(csv_file)
            prediction = 1 if score > 0.5 else 0

            # 关键修改：使用 csv_file.stem 去除 .csv 后缀
            results.append({
                "filename": csv_file.stem,  # 修改此处
                "score": score,
                "pred_score": prediction,
                "pred_label": "R" if prediction == 1 else "S"
                })
            y_pred.append(prediction)

            # 收集真实标签（如果存在）
            if csv_file.name in label_map:
                 y_true.append(label_map[csv_file.stem])

        # 保存预测结果
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_csv, index=False)
        print(f"预测结果已保存至: {output_csv}")

        # 评估与混淆矩阵（如果有标签）
        if len(y_true) > 0:
            if len(y_true) != len(y_pred):
                print(f"警告: 标签数({len(y_true)})与预测数({len(y_pred)})不匹配")
                return

            test_acc = accuracy_score(y_true, y_pred)
            print(f"\n测试集准确率: {test_acc:.2%}")

            cm_path = output_csv.parent / "Test_Confusion_Matrix.png"
            ModelVisualizer.plot_confusion_matrix(
                y_true=np.array(y_true),
                y_pred=np.array(y_pred),
                save_path=cm_path,
                style={'title': '测试集混淆矩阵'}
            )

        # 数据汇总（如果启用报告目录）
        if PathConfig.REPORT_DIR:

            # 生成测试集样本数据
            test_samples = {
                csv_file.stem: {'data': load_sample_data(csv_file)}
                for csv_file in input_dir.glob("*.csv")
            }

            # 生成测试集汇总
            test_summary_df = DataSummarizer.summarize_dataset(
                test_samples, result_df, "test"
            )

            test_summary_path = PathConfig.REPORT_DIR / "test_summary.csv"
            test_summary_df.to_csv(test_summary_path, index=False)
            print(f"\n完整数据集统计摘要已保存至: {test_summary_path}")

    def _generate_features(self, data: dict) -> list:
        """特征生成逻辑"""
        features = []
        # 菌丝特征
        hyphal_feats = HyphalFeatureEngineer.calculate(data['Cell_state'])
        features.extend([
            hyphal_feats["hyphal_germination_ratio"],
            hyphal_feats["hyphal_germinated_mean"],
            hyphal_feats["hyphal_overall_mean"],
            hyphal_feats["hyphal_germinated_std"]
        ])
        # 其他维度特征
        for dim in ['MIL', 'Cell_size']:
            base_feats = BaseFeatureEngineer.calculate(data[dim])
            features.extend([
                base_feats["mean"],
                base_feats["std"],
                base_feats["median"],
                base_feats["IQR"]
            ])
        return features



def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="药敏指数批量预测")
    parser.add_argument("-i", "--input", type=str, default=r"D:\HanX\python_data_test\Candida\score\test4",
                        help="输入目录路径（包含CSV文件）")
    parser.add_argument("-o", "--output", type=str, default=r"D:\HanX\python_data_test\Candida\score\predict4.csv",
                        help="输出结果文件路径（CSV格式）")
    parser.add_argument("-m", "--model", type=str, default=r"D:\HanX\python_data_test\Candida\score\best_model4.pkl",
                        help="模型文件路径（.pkl格式）")
    parser.add_argument("-l", "--label", type=str,default=r"D:\HanX\python_data_test\Candida\score\test_data_label4.csv",
                        help="测试集真实标签文件路径（可选）")
    args = parser.parse_args()

    # 初始化预测器
    predictor = DrugPredictor(args.model)

    # 执行批量预测
    predictor.predict_batch(
        input_dir=Path(args.input),
        output_csv=Path(args.output),
        label_path=Path(args.label) if args.label else None
    )


if __name__ == "__main__":
    main()