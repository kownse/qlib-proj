"""
TCN V1 Handler - 复现 Qlib Benchmark TCN Alpha158 配置

该 Handler 只包含 Qlib 官方 benchmark 中 FilterCol 选出的 20 个特征。

参考: qlib-src/examples/benchmarks/TCN/workflow_config_tcn_Alpha158.yaml

特征列表:
    RESI5, WVMA5, RSQR5, KLEN, RSQR10, CORR5, CORD5, CORR10,
    ROC60, RESI10, VSTD5, RSQR60, CORR60, WVMA60, STD5,
    RSQR20, CORD60, CORD10, CORR20, KLOW
"""

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.utils import get_callable_kwargs
from qlib.data.dataset import processor as processor_module
from inspect import getfullargspec


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    """
    处理处理器配置，自动添加 fit_start_time 和 fit_end_time 参数。

    从 qlib.contrib.data.handler 复制的辅助函数。
    """
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                if fit_start_time is not None and fit_end_time is not None:
                    pkwargs.update({
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    })
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


def get_tcn_v1_feature_config():
    """
    返回 Qlib benchmark TCN 选出的 20 个特征的配置。

    Returns:
        tuple: (fields, names) 其中 fields 是 Qlib 表达式列表，names 是特征名列表
    """
    fields = []
    names = []

    # K线特征 (2个)
    # KLEN: 振幅
    fields.append("($high-$low)/$open")
    names.append("KLEN")

    # KLOW: 下影线占比
    fields.append("(Less($open, $close)-$low)/$open")
    names.append("KLOW")

    # ROC (1个) - Rate of Change
    # ROC60: 60天价格变化率
    fields.append("Ref($close, 60)/$close")
    names.append("ROC60")

    # STD (1个) - 标准差
    # STD5: 5天收盘价标准差
    fields.append("Std($close, 5)/$close")
    names.append("STD5")

    # RSQR (4个) - R平方值，衡量趋势线性度
    for d in [5, 10, 20, 60]:
        fields.append(f"Rsquare($close, {d})")
        names.append(f"RSQR{d}")

    # RESI (2个) - 残差，衡量偏离趋势程度
    for d in [5, 10]:
        fields.append(f"Resi($close, {d})/$close")
        names.append(f"RESI{d}")

    # CORR (4个) - 价格与成交量的相关性
    for d in [5, 10, 20, 60]:
        fields.append(f"Corr($close, Log($volume+1), {d})")
        names.append(f"CORR{d}")

    # CORD (3个) - 价格变化与成交量变化的相关性
    for d in [5, 10, 60]:
        fields.append(f"Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})")
        names.append(f"CORD{d}")

    # VSTD (1个) - 成交量标准差
    fields.append("Std($volume, 5)/($volume+1e-12)")
    names.append("VSTD5")

    # WVMA (2个) - 成交量加权价格变化波动率
    for d in [5, 60]:
        fields.append(
            f"Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/"
            f"(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)"
        )
        names.append(f"WVMA{d}")

    return fields, names


class TCN_V1_Handler(DataHandlerLP):
    """
    TCN V1 Handler - 复现 Qlib Benchmark 的 20 特征配置

    用于与 TSDatasetH 配合，实现时间序列数据的TCN训练。

    Args:
        volatility_window: 用于标签计算的天数（默认5天）
        instruments: 股票代码列表
        start_time: 数据开始时间
        end_time: 数据结束时间
        fit_start_time: 用于拟合处理器的开始时间
        fit_end_time: 用于拟合处理器的结束时间
        infer_processors: 推断时的数据处理器列表
        learn_processors: 学习时的数据处理器列表
    """

    def __init__(
        self,
        volatility_window=5,
        instruments="csi500",
        start_time=None,
        end_time=None,
        fit_start_time=None,
        fit_end_time=None,
        infer_processors=None,
        learn_processors=None,
        **kwargs,
    ):
        self.volatility_window = volatility_window

        # 默认推断处理器 - 与 benchmark 配置一致
        if infer_processors is None:
            infer_processors = [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ]

        # 默认学习处理器
        if learn_processors is None:
            learn_processors = [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ]

        # 使用 check_transform_proc 处理处理器配置
        # 自动添加 fit_start_time 和 fit_end_time 参数
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        # 设置数据加载器配置
        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": get_tcn_v1_feature_config(),
                    "label": self.get_label_config(),
                },
                "freq": "day",
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            **kwargs,
        )

    def get_label_config(self):
        """
        生成标签配置

        使用 N 天后的收益率作为标签（与 benchmark 一致使用 Ref($close, -2)/Ref($close, -1) - 1）
        """
        # Benchmark 使用的是2天收益率，这里根据 volatility_window 调整
        label_expr = f"Ref($close, -{self.volatility_window})/Ref($close, -1) - 1"
        return [label_expr], ["LABEL0"]


# 用于直接测试
if __name__ == "__main__":
    fields, names = get_tcn_v1_feature_config()
    print(f"Total features: {len(names)}")
    print("\nFeatures:")
    for i, (field, name) in enumerate(zip(fields, names), 1):
        print(f"  {i:2d}. {name:10s}: {field}")
