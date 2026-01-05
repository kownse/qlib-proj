"""
运行 Temporal Fusion Transformer (TFT) 模型，预测N天股票价格波动率

波动率定义：未来N个交易日波动变化

扩展特征：包含 Alpha158 默认指标 + TA-Lib 技术指标
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
script_dir = Path(__file__).parent.parent  # scripts directory
sys.path.insert(0, str(script_dir))

from pathlib import Path
import argparse
import os
import sys
import datetime as dte
from typing import Union

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

import qlib
from qlib.constant import REG_US
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import ModelFT

# Disable TF2 behavior
tf.disable_v2_behavior()

# Import TA-Lib custom operators
from utils.talib_ops import TALIB_OPS

# Import extended data handlers
from data.datahandler_ext import Alpha158_Volatility, Alpha158_Volatility_TALib
from utils.utils import evaluate_model

# Add TFT benchmark path for imports
TFT_PATH = Path(__file__).parent.parent / "qlib" / "examples" / "benchmarks" / "TFT"
sys.path.insert(0, str(TFT_PATH))

import data_formatters.base
import libs.tft_model
import libs.utils as tft_utils
import sklearn.preprocessing


# ========== 配置 ==========

# 数据路径
PROJECT_ROOT = Path(__file__).parent.parent.parent  # 项目根目录
QLIB_DATA_PATH = PROJECT_ROOT / "my_data" / "qlib_us"

# 股票池
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ"]

# 时间划分
TRAIN_START = "2015-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# 波动率预测窗口（天数）
VOLATILITY_WINDOW = 2


# ========== TFT 特征选择 ==========
# 从 Alpha158_Volatility_TALib 中选择最重要的特征用于 TFT
# TFT 对特征数量有限制，选择约 20 个最具代表性的特征
SELECTED_FEATURES = [
    # 基础价格特征
    "KMID", "KLEN", "KMID2", "KLOW", "KLOW2",
    # 动量类
    "ROC5", "ROC10", "ROC20", "ROC60",
    # 波动率类
    "STD5", "STD10", "STD20", "STD60",
    # 相关性类
    "CORR5", "CORR10", "CORR20", "CORR60",
    # 残差类
    "RESI5", "RESI10",
    # R-squared
    "RSQR5", "RSQR10",
]

# 扩展 TA-Lib 特征（如果使用 TA-Lib）
TALIB_SELECTED_FEATURES = [
    "TALIB_RSI14",
    "TALIB_ATR14",
    "TALIB_ADX14",
    "TALIB_MACD",
    "TALIB_BB_WIDTH20",
    "TALIB_MOM10",
    "TALIB_CCI14",
    "TALIB_WILLR14",
]


# ========== Data Formatter for Alpha158_Volatility_TALib ==========

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class Alpha158VolatilityTALibFormatter(GenericDataFormatter):
    """Data formatter for Alpha158_Volatility_TALib dataset."""

    def __init__(self, feature_cols, use_talib=False):
        """Initialize formatter with selected features."""
        self.feature_cols = feature_cols
        self.use_talib = use_talib
        self._build_column_definition()

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def _build_column_definition(self):
        """Build column definition based on selected features."""
        self._column_definition = [
            ("instrument", DataTypes.CATEGORICAL, InputTypes.ID),
            ("LABEL0", DataTypes.REAL_VALUED, InputTypes.TARGET),
            ("date", DataTypes.DATE, InputTypes.TIME),
            ("month", DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
            ("day_of_week", DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ]

        # Add selected features as observed inputs
        for feat in self.feature_cols:
            self._column_definition.append(
                (feat, DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT)
            )

        # Add static input
        self._column_definition.append(
            ("const", DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
        )

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied."""
        print("Setting scalers with training data...")

        column_definitions = self.get_column_definition()
        id_column = tft_utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = tft_utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

        self.identifiers = list(df[id_column].unique())

        real_inputs = tft_utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        data = df[real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
            df[[target_column]].values
        )

        categorical_inputs = tft_utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        """Performs feature transformations."""
        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError("Scalers have not been set!")

        column_definitions = self.get_column_definition()

        real_inputs = tft_utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )
        categorical_inputs = tft_utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions, {InputTypes.ID, InputTypes.TIME}
        )

        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output

    def format_predictions(self, predictions):
        """Reverts any normalisation to give predictions in original scale."""
        output = predictions.copy()
        column_names = predictions.columns

        for col in column_names:
            if col not in {"forecast_time", "identifier"}:
                output[col] = self._target_scaler.inverse_transform(predictions[[col]])

        return output

    def get_fixed_params(self):
        """Returns fixed model parameters for experiments."""
        return {
            "total_time_steps": 6 + 6,
            "num_encoder_steps": 6,
            "num_epochs": 50,  # Reduced for faster training
            "early_stopping_patience": 10,
            "multiprocessing_workers": 4,
        }

    def get_default_model_params(self):
        """Returns default optimised model parameters."""
        return {
            "dropout_rate": 0.3,
            "hidden_layer_size": 128,
            "learning_rate": 0.001,
            "minibatch_size": 64,
            "max_gradient_norm": 0.01,
            "num_heads": 4,
            "stack_size": 1,
        }


# ========== Helper Functions ==========

def get_shifted_label(data_df, shifts=5, col_shift="LABEL0"):
    """Shift labels for TFT's multi-step prediction."""
    return data_df[[col_shift]].groupby("instrument", group_keys=False).apply(
        lambda df: df.shift(shifts)
    )


def fill_test_na(test_df):
    """Fill NA values in test data with group mean."""
    test_df_res = test_df.copy()
    feature_cols = ~test_df_res.columns.str.contains("label", case=False)
    test_feature_fna = (
        test_df_res.loc[:, feature_cols]
        .groupby("datetime", group_keys=False)
        .apply(lambda df: df.fillna(df.mean()))
    )
    test_df_res.loc[:, feature_cols] = test_feature_fna
    return test_df_res


def process_qlib_data(df, feature_cols, fillna=False):
    """Prepare data to fit the TFT model."""
    label_col = ["LABEL0"]

    # Filter to only selected features + label
    available_features = [f for f in feature_cols if f in df.columns]
    temp_df = df.loc[:, available_features + label_col].copy()

    if fillna:
        temp_df = fill_test_na(temp_df)

    temp_df = temp_df.swaplevel()
    temp_df = temp_df.sort_index()
    temp_df = temp_df.reset_index(level=0)

    dates = pd.to_datetime(temp_df.index)
    temp_df["date"] = dates
    temp_df["day_of_week"] = dates.dayofweek
    temp_df["month"] = dates.month
    temp_df["year"] = dates.year
    temp_df["const"] = 1.0

    return temp_df


def process_predicted(df, col_name):
    """Transform the TFT predicted data into Qlib format."""
    df_res = df.copy()
    df_res = df_res.rename(columns={
        "forecast_time": "datetime",
        "identifier": "instrument",
        "t+4": col_name
    })
    df_res = df_res.set_index(["datetime", "instrument"]).sort_index()
    df_res = df_res[[col_name]]
    return df_res


def format_score(forecast_df, col_name="pred", label_shift=5):
    """Format prediction scores."""
    pred = process_predicted(forecast_df, col_name=col_name)
    pred = get_shifted_label(pred, shifts=-label_shift, col_shift=col_name)
    pred = pred.dropna()[col_name]
    return pred


def transform_df(df, col_name="LABEL0"):
    """Transform dataset output to DataFrame."""
    df_res = df["feature"].copy()
    df_res[col_name] = df["label"]
    return df_res


# ========== TFT Model for Volatility ==========

class TFTVolatilityModel(ModelFT):
    """TFT Model for Volatility Prediction."""

    def __init__(self, feature_cols, use_talib=False, label_shift=5, **kwargs):
        self.model = None
        self.feature_cols = feature_cols
        self.use_talib = use_talib
        self.label_shift = label_shift
        self.params = kwargs

    def _prepare_data(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        return transform_df(df_train), transform_df(df_valid)

    def fit(self, dataset: DatasetH, model_folder="tft_volatility_model", gpu_id=0, **kwargs):
        dtrain, dvalid = self._prepare_data(dataset)

        # Shift labels for multi-step prediction
        dtrain.loc[:, "LABEL0"] = get_shifted_label(dtrain, shifts=self.label_shift, col_shift="LABEL0")
        dvalid.loc[:, "LABEL0"] = get_shifted_label(dvalid, shifts=self.label_shift, col_shift="LABEL0")

        train = process_qlib_data(dtrain, self.feature_cols, fillna=True).dropna()
        valid = process_qlib_data(dvalid, self.feature_cols, fillna=True).dropna()

        print(f"    Training data shape: {train.shape}")
        print(f"    Validation data shape: {valid.shape}")

        # Create data formatter
        available_features = [f for f in self.feature_cols if f in train.columns]
        self.data_formatter = Alpha158VolatilityTALibFormatter(available_features, self.use_talib)
        self.model_folder = model_folder
        self.gpu_id = gpu_id

        use_gpu = (True, self.gpu_id)

        ModelClass = libs.tft_model.TemporalFusionTransformer

        default_keras_session = tf.keras.backend.get_session()

        if use_gpu[0]:
            self.tf_config = tft_utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=use_gpu[1])
        else:
            self.tf_config = tft_utils.get_default_tensorflow_config(tf_device="cpu")

        self.data_formatter.set_scalers(train)

        fixed_params = self.data_formatter.get_fixed_params()
        params = self.data_formatter.get_default_model_params()
        params = {**params, **fixed_params}

        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        params["model_folder"] = self.model_folder

        print("\n*** Begin TFT Training ***")
        print(f"    Model parameters:")
        for key, value in params.items():
            print(f"      - {key}: {value}")

        tf.reset_default_graph()

        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.sess = tf.Session(config=self.tf_config)
            tf.keras.backend.set_session(self.sess)
            self.model = ModelClass(params, use_cudnn=use_gpu[0])
            self.sess.run(tf.global_variables_initializer())
            self.model.fit(train_df=train, valid_df=valid)
            print("*** Finished TFT Training ***")

            saved_model_dir = os.path.join(self.model_folder, "saved_model")
            if not os.path.exists(saved_model_dir):
                os.makedirs(saved_model_dir)
            self.model.save(saved_model_dir)

            tf.keras.backend.set_session(default_keras_session)

        print(f"Training completed at {dte.datetime.now()}.")

    def predict(self, dataset):
        if self.model is None:
            raise ValueError("Model is not fitted yet!")

        d_test = dataset.prepare("test", col_set=["feature", "label"])
        d_test = transform_df(d_test)
        d_test.loc[:, "LABEL0"] = get_shifted_label(d_test, shifts=self.label_shift, col_shift="LABEL0")
        test = process_qlib_data(d_test, self.feature_cols, fillna=True).dropna()

        print("\n*** Begin TFT Prediction ***")
        default_keras_session = tf.keras.backend.get_session()

        with self.tf_graph.as_default():
            tf.keras.backend.set_session(self.sess)
            output_map = self.model.predict(test, return_targets=True)
            p50_forecast = self.data_formatter.format_predictions(output_map["p50"])
            p90_forecast = self.data_formatter.format_predictions(output_map["p90"])
            tf.keras.backend.set_session(default_keras_session)

        predict50 = format_score(p50_forecast, "pred", 1)
        predict90 = format_score(p90_forecast, "pred", 1)
        predict = (predict50 + predict90) / 2

        print("*** Finished TFT Prediction ***")
        return predict

    def finetune(self, dataset: DatasetH):
        pass

    def to_pickle(self, path: Union[Path, str]):
        drop_attrs = ["model", "tf_graph", "sess", "data_formatter"]
        orig_attr = {}
        for attr in drop_attrs:
            orig_attr[attr] = getattr(self, attr, None)
            setattr(self, attr, None)
        super(TFTVolatilityModel, self).to_pickle(path)
        for attr in drop_attrs:
            setattr(self, attr, orig_attr[attr])


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description='TFT Stock Price Volatility Prediction')
    parser.add_argument('--nday', type=int, default=2,
                        help='Volatility prediction window in days (default: 2)')
    parser.add_argument('--use-talib', action='store_true',
                        help='Use extended TA-Lib features (default: False)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    args = parser.parse_args()

    global VOLATILITY_WINDOW
    VOLATILITY_WINDOW = args.nday

    print("=" * 70)
    print(f"TFT {VOLATILITY_WINDOW}-Day Stock Price Volatility Prediction")
    if args.use_talib:
        print("Features: Alpha158 + TA-Lib Technical Indicators")
    else:
        print("Features: Alpha158 (default)")
    print("=" * 70)

    # 1. Initialize Qlib
    print("\n[1] Initializing Qlib...")
    if args.use_talib:
        qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US, custom_ops=TALIB_OPS)
        print("    Qlib initialized with TA-Lib custom operators")
    else:
        qlib.init(provider_uri=str(QLIB_DATA_PATH), region=REG_US)
        print("    Qlib initialized")

    # 2. Check data availability
    print("\n[2] Checking data availability...")
    instruments = D.instruments(market="all")
    available_instruments = list(D.list_instruments(instruments))
    print(f"    Available instruments: {len(available_instruments)}")

    test_df = D.features(
        instruments=["AAPL"],
        fields=["$close", "$volume"],
        start_time=TEST_START,
        end_time=TEST_END
    )
    print(f"    AAPL sample data shape: {test_df.shape}")
    print(f"    Date range: {test_df.index.get_level_values('datetime').min().date()} to "
          f"{test_df.index.get_level_values('datetime').max().date()}")

    # 3. Create DataHandler
    print(f"\n[3] Creating DataHandler with {VOLATILITY_WINDOW}-day volatility label...")

    # Select features based on TA-Lib flag
    if args.use_talib:
        print(f"    Features: Alpha158 + TA-Lib (~300+ technical indicators)")
        feature_cols = SELECTED_FEATURES + TALIB_SELECTED_FEATURES
        handler = Alpha158_Volatility_TALib(
            volatility_window=VOLATILITY_WINDOW,
            instruments=TEST_SYMBOLS,
            start_time=TRAIN_START,
            end_time=TEST_END,
            fit_start_time=TRAIN_START,
            fit_end_time=TRAIN_END,
            infer_processors=[],
        )
    else:
        print(f"    Features: Alpha158 (158 technical indicators)")
        feature_cols = SELECTED_FEATURES
        handler = Alpha158_Volatility(
            volatility_window=VOLATILITY_WINDOW,
            instruments=TEST_SYMBOLS,
            start_time=TRAIN_START,
            end_time=TEST_END,
            fit_start_time=TRAIN_START,
            fit_end_time=TRAIN_END,
            infer_processors=[],
        )

    print(f"    Label: {VOLATILITY_WINDOW}-day realized volatility")
    print("    DataHandler created")

    # 4. Create Dataset
    print("\n[4] Creating Dataset...")
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (TRAIN_START, TRAIN_END),
            "valid": (VALID_START, VALID_END),
            "test": (TEST_START, TEST_END),
        }
    )

    train_data = dataset.prepare("train", col_set="feature")
    print(f"    Train features shape: {train_data.shape}")
    print(f"      (samples x features)")

    # Display available features
    print(f"\n[5] Selected features for TFT:")
    available_features = [f for f in feature_cols if f in train_data.columns]
    print(f"    Using {len(available_features)} features:")
    for i, feat in enumerate(available_features[:10]):
        print(f"      {i+1}. {feat}")
    if len(available_features) > 10:
        print(f"      ... and {len(available_features) - 10} more")

    # 5. Analyze label distribution
    print("\n[6] Analyzing label distribution...")
    train_label = dataset.prepare("train", col_set="label")
    valid_label = dataset.prepare("valid", col_set="label")

    print(f"    Train set volatility statistics:")
    print(f"      Mean:   {train_label['LABEL0'].mean():.4f}")
    print(f"      Std:    {train_label['LABEL0'].std():.4f}")
    print(f"      Median: {train_label['LABEL0'].median():.4f}")
    print(f"      Min:    {train_label['LABEL0'].min():.4f}")
    print(f"      Max:    {train_label['LABEL0'].max():.4f}")

    print(f"\n    Valid set volatility statistics:")
    print(f"      Mean:   {valid_label['LABEL0'].mean():.4f}")
    print(f"      Std:    {valid_label['LABEL0'].std():.4f}")

    # 6. Train TFT model
    print("\n[7] Training Temporal Fusion Transformer (TFT) model...")

    model_folder = str(PROJECT_ROOT / "my_models" / f"tft_volatility_{VOLATILITY_WINDOW}d")

    model = TFTVolatilityModel(
        feature_cols=available_features,
        use_talib=args.use_talib,
        label_shift=5,
    )

    model.fit(
        dataset,
        model_folder=model_folder,
        gpu_id=args.gpu_id,
    )
    print("    Model training completed")

    # 7. Predict
    print("\n[8] Generating predictions...")
    pred = model.predict(dataset)

    test_pred = pred.loc[TEST_START:TEST_END]
    print(f"    Prediction shape: {test_pred.shape}")
    print(f"    Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    # 8. Evaluate
    evaluate_model(dataset, test_pred, PROJECT_ROOT, VOLATILITY_WINDOW)


if __name__ == "__main__":
    main()
