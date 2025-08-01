# 路径配置

from pathlib import Path


class PathConfig:
    # 自动获取项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent

    # 新的输入文件夹和标签路径
    DATA_DIR = Path(r"D:\HanX\python_data_test\Candida\score\train4")  # 修改为新的输入路径
    LABEL_PATH = Path(r"D:\HanX\python_data_test\Candida\score\trains_data_label4.csv")  # 修改为新的标签路径

    # 新的模型和报告输出路径
    MODEL_DIR = Path(r"D:\HanX\python_data_test\Candida\score")  # 修改为新的模型存储路径
    REPORT_DIR = Path(r"D:\HanX\python_data_test\Candida\score")  # 修改为新的报告存储路径

    @classmethod
    def setup_dirs(cls):
        """创建必要目录"""
        cls.MODEL_DIR.mkdir(exist_ok=True)
        cls.REPORT_DIR.mkdir(exist_ok=True)