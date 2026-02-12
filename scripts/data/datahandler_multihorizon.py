"""
Multi-Horizon Data Handlers

为 multi-horizon 训练提供多个时间窗口的标签。
例如同时生成 2天、5天、10天 的 forward return 标签，
用于 multi-task learning 中的辅助目标。

标签列:
  - LABEL_2d: 2天 forward return
  - LABEL_5d: 5天 forward return (主目标)
  - LABEL_10d: 10天 forward return
"""

import sys
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

PROJECT_ROOT = script_dir.parent
DEFAULT_MACRO_PATH = PROJECT_ROOT / "my_data" / "macro_processed" / "macro_features.parquet"

from qlib.contrib.data.handler import Alpha158, check_transform_proc, _DEFAULT_LEARN_PROCESSORS
from qlib.data.dataset.handler import DataHandlerLP


class Alpha158_MultiHorizon(Alpha158):
    """
    Alpha158 特征 + 多个时间窗口的 forward return 标签。

    默认标签: 2d, 5d, 10d forward returns。
    主目标通过 primary_horizon 指定（默认5天）。
    """

    def __init__(self, horizons=None, primary_horizon=5, **kwargs):
        """
        Args:
            horizons: 预测时间窗口列表，如 [2, 5, 10]
            primary_horizon: 主要预测目标的天数
            **kwargs: 传递给父类的参数
        """
        self.horizons = horizons or [2, 5, 10]
        self.primary_horizon = primary_horizon
        if primary_horizon not in self.horizons:
            self.horizons.append(primary_horizon)
            self.horizons.sort()
        super().__init__(**kwargs)

    def get_label_config(self):
        """返回多个 horizon 的标签配置"""
        fields = []
        names = []
        for h in self.horizons:
            expr = f"Ref($close, -{h})/Ref($close, -1) - 1"
            fields.append(expr)
            names.append(f"LABEL_{h}d")
        return fields, names


class Alpha158_TALib_Lite_MultiHorizon(DataHandlerLP):
    """
    Alpha158 + 精选 TA-Lib 指标 + 多个时间窗口的 forward return 标签。

    复用 Alpha158_Volatility_TALib_Lite 的特征配置，
    标签改为多个 horizon 的 forward returns。
    """

    def __init__(
        self,
        horizons=None,
        primary_horizon=5,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        self.horizons = horizons or [2, 5, 10]
        self.primary_horizon = primary_horizon
        if primary_horizon not in self.horizons:
            self.horizons.append(primary_horizon)
            self.horizons.sort()

        if learn_processors is None:
            learn_processors = _DEFAULT_LEARN_PROCESSORS

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        """复用 Alpha158_Volatility_TALib_Lite 的特征配置"""
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW"],
            },
            "rolling": {
                "exclude": ["VMA", "VSTD"],
            },
        }
        fields, names = Alpha158.get_feature_config(conf)

        talib_fields, talib_names = self._get_talib_features()
        fields.extend(talib_fields)
        names.extend(talib_names)

        return fields, names

    def _get_talib_features(self):
        """精选 TA-Lib 指标（与 Alpha158_Volatility_TALib_Lite 一致）"""
        fields = []
        names = []

        # 动量
        fields.append("TALIB_RSI($close, 14)")
        names.append("TALIB_RSI14")
        fields.append("TALIB_MOM($close, 10)/$close")
        names.append("TALIB_MOM10")
        fields.append("TALIB_ROC($close, 10)")
        names.append("TALIB_ROC10")
        fields.append("TALIB_CMO($close, 14)")
        names.append("TALIB_CMO14")

        # MACD
        fields.append("TALIB_MACD_MACD($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD")
        fields.append("TALIB_MACD_SIGNAL($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_SIGNAL")
        fields.append("TALIB_MACD_HIST($close, 12, 26, 9)/$close")
        names.append("TALIB_MACD_HIST")

        # 移动平均
        fields.append("TALIB_EMA($close, 20)/$close")
        names.append("TALIB_EMA20")
        fields.append("TALIB_SMA($close, 20)/$close")
        names.append("TALIB_SMA20")

        # 布林带
        fields.append("(TALIB_BBANDS_UPPER($close, 20, 2) - $close)/$close")
        names.append("TALIB_BB_UPPER_DIST")
        fields.append("($close - TALIB_BBANDS_LOWER($close, 20, 2))/$close")
        names.append("TALIB_BB_LOWER_DIST")
        fields.append("(TALIB_BBANDS_UPPER($close, 20, 2) - TALIB_BBANDS_LOWER($close, 20, 2))/$close")
        names.append("TALIB_BB_WIDTH")

        # 波动率
        fields.append("TALIB_ATR($high, $low, $close, 14)/$close")
        names.append("TALIB_ATR14")
        fields.append("TALIB_NATR($high, $low, $close, 14)")
        names.append("TALIB_NATR14")

        # 趋势
        fields.append("TALIB_ADX($high, $low, $close, 14)")
        names.append("TALIB_ADX14")
        fields.append("TALIB_PLUS_DI($high, $low, $close, 14)")
        names.append("TALIB_PLUS_DI14")
        fields.append("TALIB_MINUS_DI($high, $low, $close, 14)")
        names.append("TALIB_MINUS_DI14")

        # 随机
        fields.append("TALIB_STOCH_K($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_K")
        fields.append("TALIB_STOCH_D($high, $low, $close, 5, 3, 3)")
        names.append("TALIB_STOCH_D")

        # 统计
        fields.append("TALIB_STDDEV($close, 20, 1)/$close")
        names.append("TALIB_STDDEV20")

        return fields, names

    def get_label_config(self):
        """返回多个 horizon 的标签配置"""
        fields = []
        names = []
        for h in self.horizons:
            expr = f"Ref($close, -{h})/Ref($close, -1) - 1"
            fields.append(expr)
            names.append(f"LABEL_{h}d")
        return fields, names


class Alpha158_TALib_Lite_Macro_MultiHorizon(Alpha158_TALib_Lite_MultiHorizon):
    """
    Alpha158 + 精选 TA-Lib 指标 + Macro 特征 + 多个时间窗口 forward return 标签。

    继承 Alpha158_TALib_Lite_MultiHorizon 的全部特征和多标签配置，
    在 process_data() 后注入 macro 特征（与 Alpha158_Volatility_TALib_Macro 相同方式）。

    总特征: ~170 (Alpha158+TA-Lib) + ~23 (core macro) 或 ~105 (all macro)
    """

    # Macro 特征列表（与 Alpha158_Volatility_TALib_Macro 一致）
    ALL_MACRO_FEATURES = [
        # VIX (8)
        "macro_vix_level", "macro_vix_zscore20",
        "macro_vix_pct_1d", "macro_vix_pct_5d", "macro_vix_pct_10d",
        "macro_vix_ma5_ratio", "macro_vix_ma20_ratio", "macro_vix_regime",
        # VIX Term Structure (5)
        "macro_vix_term_structure", "macro_vix_contango", "macro_vix_term_zscore",
        "macro_uvxy_pct_5d", "macro_svxy_pct_5d",
        # Gold (5)
        "macro_gld_pct_1d", "macro_gld_pct_5d", "macro_gld_pct_20d",
        "macro_gld_ma20_ratio", "macro_gld_vol20",
        # Bond (8)
        "macro_tlt_pct_1d", "macro_tlt_pct_5d", "macro_tlt_pct_20d",
        "macro_tlt_ma20_ratio", "macro_yield_curve", "macro_yield_curve_chg5",
        "macro_ief_pct_5d", "macro_bond_vol20",
        # Dollar (4)
        "macro_uup_pct_1d", "macro_uup_pct_5d",
        "macro_uup_ma20_ratio", "macro_uup_strength",
        # Oil (4)
        "macro_uso_pct_1d", "macro_uso_pct_5d", "macro_uso_pct_20d",
        "macro_uso_vol20",
        # Sector ETFs (33 = 11 sectors x 3 features)
        "macro_xlk_pct_5d", "macro_xlk_pct_20d", "macro_xlk_vs_spy",
        "macro_xlf_pct_5d", "macro_xlf_pct_20d", "macro_xlf_vs_spy",
        "macro_xle_pct_5d", "macro_xle_pct_20d", "macro_xle_vs_spy",
        "macro_xlv_pct_5d", "macro_xlv_pct_20d", "macro_xlv_vs_spy",
        "macro_xli_pct_5d", "macro_xli_pct_20d", "macro_xli_vs_spy",
        "macro_xlp_pct_5d", "macro_xlp_pct_20d", "macro_xlp_vs_spy",
        "macro_xly_pct_5d", "macro_xly_pct_20d", "macro_xly_vs_spy",
        "macro_xlu_pct_5d", "macro_xlu_pct_20d", "macro_xlu_vs_spy",
        "macro_xlre_pct_5d", "macro_xlre_pct_20d", "macro_xlre_vs_spy",
        "macro_xlb_pct_5d", "macro_xlb_pct_20d", "macro_xlb_vs_spy",
        "macro_xlc_pct_5d", "macro_xlc_pct_20d", "macro_xlc_vs_spy",
        # Credit/Risk (8)
        "macro_hyg_pct_5d", "macro_hyg_pct_20d", "macro_hyg_vs_lqd",
        "macro_hyg_lqd_chg5", "macro_jnk_vol20", "macro_credit_stress",
        "macro_hyg_tlt_ratio", "macro_hyg_tlt_chg5",
        # Global (8)
        "macro_eem_pct_5d", "macro_eem_pct_20d", "macro_eem_vs_spy",
        "macro_efa_pct_5d", "macro_efa_vs_spy",
        "macro_fxi_pct_5d", "macro_ewj_pct_5d", "macro_global_risk",
        # Benchmark (6)
        "macro_spy_pct_1d", "macro_spy_pct_5d", "macro_spy_pct_20d",
        "macro_spy_vol20", "macro_qqq_vs_spy", "macro_spy_ma20_ratio",
        # Treasury Yields (10)
        "macro_yield_2y", "macro_yield_10y", "macro_yield_30y",
        "macro_yield_2s10s", "macro_yield_3m10y",
        "macro_yield_10y_chg5", "macro_yield_10y_chg20",
        "macro_yield_curve_slope", "macro_yield_curve_zscore", "macro_yield_inversion",
        # FRED Credit Spreads (5)
        "macro_hy_spread", "macro_hy_spread_zscore", "macro_hy_spread_chg5",
        "macro_ig_spread", "macro_credit_risk",
        # Cross-asset (5)
        "macro_risk_on_off", "macro_gold_oil_ratio", "macro_gold_oil_ratio_chg",
        "macro_stock_bond_corr", "macro_market_stress",
    ]

    CORE_MACRO_FEATURES = [
        "macro_vix_level", "macro_vix_zscore20", "macro_vix_pct_5d", "macro_vix_regime",
        "macro_vix_term_structure",
        "macro_gld_pct_5d", "macro_tlt_pct_5d", "macro_yield_curve",
        "macro_uup_pct_5d", "macro_uso_pct_5d",
        "macro_spy_pct_5d", "macro_spy_vol20",
        "macro_hyg_vs_lqd", "macro_credit_stress",
        "macro_eem_vs_spy", "macro_global_risk",
        "macro_yield_10y", "macro_yield_2s10s", "macro_yield_inversion",
        "macro_hy_spread", "macro_hy_spread_zscore",
        "macro_risk_on_off", "macro_market_stress",
    ]

    VIX_ONLY_FEATURES = [
        "macro_vix_level", "macro_vix_zscore20",
        "macro_vix_pct_1d", "macro_vix_pct_5d", "macro_vix_pct_10d",
        "macro_vix_ma5_ratio", "macro_vix_ma20_ratio", "macro_vix_regime",
        "macro_vix_term_structure", "macro_vix_contango", "macro_vix_term_zscore",
        "macro_uvxy_pct_5d", "macro_svxy_pct_5d",
    ]

    def __init__(
        self,
        horizons=None,
        primary_horizon=5,
        macro_data_path: Union[str, Path] = None,
        macro_features: str = "core",
        **kwargs,
    ):
        """
        Args:
            horizons: 预测时间窗口列表
            primary_horizon: 主要预测目标的天数
            macro_data_path: macro 特征 parquet 文件路径
            macro_features: macro 特征集 ("all", "core", "vix_only", "none")
            **kwargs: 传递给父类的参数
        """
        self.macro_data_path = Path(macro_data_path) if macro_data_path else DEFAULT_MACRO_PATH
        self.macro_features = macro_features
        self._macro_df = self._load_macro_features()

        super().__init__(horizons=horizons, primary_horizon=primary_horizon, **kwargs)

    def process_data(self, with_fit: bool = False):
        """在父类处理完后注入 macro 特征"""
        super().process_data(with_fit=with_fit)

        if self._macro_df is not None and self.macro_features != "none":
            self._add_macro_to_processed_data()

    def _add_macro_to_processed_data(self):
        """将 macro 特征添加到 _learn 和 _infer"""
        try:
            macro_cols = self._get_macro_feature_columns()
            available_cols = [c for c in macro_cols if c in self._macro_df.columns]

            if not available_cols:
                return

            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._merge_macro_to_df(self._learn, available_cols)
                print(f"Added {len(available_cols)} macro features to learn data")

            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._merge_macro_to_df(self._infer, available_cols)

        except Exception as e:
            print(f"Warning: Error adding macro features: {e}")

    def _merge_macro_to_df(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """将 macro 特征合并到 DataFrame"""
        datetime_col = df.index.names[0]
        main_datetimes = df.index.get_level_values(datetime_col)
        has_multi_columns = isinstance(df.columns, pd.MultiIndex)

        macro_data = {}
        for col in cols:
            macro_series = self._macro_df[col]
            aligned_values = macro_series.reindex(main_datetimes).values

            if has_multi_columns:
                macro_data[('feature', col)] = aligned_values
            else:
                macro_data[col] = aligned_values

        macro_df = pd.DataFrame(macro_data, index=df.index)
        merged = pd.concat([df, macro_df], axis=1, copy=False)
        return merged.copy()

    def _load_macro_features(self) -> Optional[pd.DataFrame]:
        """加载 macro 特征"""
        if self.macro_features == "none":
            return None

        if not self.macro_data_path.exists():
            print(f"Warning: Macro features file not found: {self.macro_data_path}")
            return None

        try:
            df = pd.read_parquet(self.macro_data_path)
            print(f"Loaded macro features: {df.shape}, "
                  f"date range: {df.index.min()} to {df.index.max()}")
            return df
        except Exception as e:
            print(f"Warning: Failed to load macro features: {e}")
            return None

    def _get_macro_feature_columns(self) -> List[str]:
        """根据配置返回 macro 特征列"""
        if self.macro_features == "core":
            return self.CORE_MACRO_FEATURES
        elif self.macro_features == "vix_only":
            return self.VIX_ONLY_FEATURES
        elif self.macro_features == "none":
            return []
        else:  # "all"
            return self.ALL_MACRO_FEATURES
