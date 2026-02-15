#!/usr/bin/env python
"""
Daily Trading Common Utilities

Common functions shared between run_daily_trading.py and run_daily_trading_ensemble.py
"""

import os
import sys
import subprocess
import smtplib
import pickle
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import numpy as np

# 设置环境变量（避免 TA-Lib 多线程问题）
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置路径
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# 确保 SCRIPTS_DIR 在 sys.path 中
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def send_email(subject: str, body: str, to_email: str, html_body: str = None) -> bool:
    """
    发送邮件

    Parameters
    ----------
    subject : str
        邮件主题
    body : str
        邮件正文（纯文本）
    to_email : str
        收件人邮箱
    html_body : str, optional
        HTML 格式的邮件正文

    Returns
    -------
    bool
        发送是否成功
    """
    smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
    smtp_port = int(os.environ.get('SMTP_PORT', 587))
    smtp_user = os.environ.get('SMTP_USER', '')
    smtp_password = os.environ.get('SMTP_PASSWORD', '')

    if not smtp_user or not smtp_password:
        print("Warning: SMTP credentials not configured. Email not sent.")
        print("Please set SMTP_USER and SMTP_PASSWORD in .env file")
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = to_email

        part1 = MIMEText(body, 'plain', 'utf-8')
        msg.attach(part1)

        if html_body:
            part2 = MIMEText(html_body, 'html', 'utf-8')
            msg.attach(part2)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())

        print(f"Email sent successfully to {to_email}")
        return True

    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


def format_trading_details_html(trading_details: list, model_info: str = "") -> str:
    """将交易详情格式化为 HTML"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    title = f"Daily Trading Report ({model_info})" if model_info else "Daily Trading Report"
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .day-header {{ background-color: #4CAF50; color: white; padding: 10px; margin-top: 20px; }}
            .holdings {{ background-color: #f9f9f9; padding: 10px; }}
            .buy {{ color: green; font-weight: bold; }}
            .sell {{ color: red; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>Generated at: {timestamp}</p>
    """

    for day in trading_details:
        return_class = 'positive' if day.get('return', 0) >= 0 else 'negative'
        html += f"""
        <div class="day-header">
            <strong>Date: {day['date']}</strong> |
            Total Value: ${day['total_value']:,.2f} |
            Cash: ${day['cash']:,.2f} |
            Daily Return: <span class="{return_class}">{day['return']:.2%}</span>
        </div>
        """

        if day.get('sells'):
            html += '<div class="sell">SELL:</div><ul>'
            for sell in day['sells']:
                html += f"<li>{sell['stock'].upper()}: {sell['amount']:.0f} shares</li>"
            html += '</ul>'

        if day.get('buys'):
            html += '<div class="buy">BUY:</div><ul>'
            for buy in day['buys']:
                html += f"<li>{buy['stock'].upper()}: {buy['amount']:.0f} shares (total: {buy['total']:.0f})</li>"
            html += '</ul>'

        if day.get('holdings'):
            html += '<div class="holdings"><strong>HOLDINGS:</strong>'
            html += '<table><tr><th>Symbol</th><th>Shares</th><th>Value</th></tr>'
            for holding in day['holdings']:
                html += f"<tr><td>{holding['stock'].upper()}</td><td>{holding['amount']:.0f}</td><td>${holding['value']:,.2f}</td></tr>"
            html += '</table></div>'

    html += """
    </body>
    </html>
    """
    return html


def format_trading_details_text(trading_details: list, model_info: str = "") -> str:
    """将交易详情格式化为纯文本"""
    lines = []
    lines.append("=" * 70)
    title = f"DAILY TRADING REPORT ({model_info})" if model_info else "DAILY TRADING REPORT"
    lines.append(title)
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    for day in trading_details:
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"Date: {day['date']}  |  Total Value: ${day['total_value']:,.2f}  |  Cash: ${day['cash']:,.2f}  |  Daily Return: {day['return']:.2%}")

        if day.get('sells'):
            lines.append("  SELL:")
            for sell in day['sells']:
                lines.append(f"    {sell['stock'].upper()}: {sell['amount']:.0f} shares")

        if day.get('buys'):
            lines.append("  BUY:")
            for buy in day['buys']:
                lines.append(f"    {buy['stock'].upper()}: {buy['amount']:.0f} shares (total: {buy['total']:.0f})")

        if day.get('holdings'):
            lines.append(f"  HOLDINGS ({len(day['holdings'])} stocks):")
            for holding in day['holdings']:
                lines.append(f"    {holding['stock'].upper()}: {holding['amount']:.0f} shares (${holding['value']:,.2f})")

    return '\n'.join(lines)


def run_command(cmd: list, description: str) -> bool:
    """运行命令并显示输出"""
    print(f"\n{'='*60}")
    print(f"[STEP] {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def get_latest_data_date() -> str:
    """获取数据的最新日期"""
    csv_dir = PROJECT_ROOT / "my_data" / "csv_us"
    latest_date = None

    for symbol in ['AAPL', 'MSFT', 'SPY']:
        csv_path = csv_dir / f"{symbol}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
            if not df.empty:
                stock_latest = df.index.max()
                if latest_date is None or stock_latest > latest_date:
                    latest_date = stock_latest

    if latest_date is not None:
        return latest_date.strftime("%Y-%m-%d")
    return datetime.now().strftime("%Y-%m-%d")


def get_latest_calendar_dates() -> tuple:
    """获取 Qlib 日历的最新交易日和倒数第二个交易日

    Returns:
        (latest_date, second_last_date): 最新日期和倒数第二个日期
        Qlib 回测需要访问 end_date 的下一天，所以实际可用的最后一天是倒数第二天
    """
    calendar_path = PROJECT_ROOT / "my_data" / "qlib_us" / "calendars" / "day.txt"
    if calendar_path.exists():
        with open(calendar_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) >= 2:
                return lines[-1], lines[-2]
            elif len(lines) == 1:
                return lines[-1], lines[-1]
    today = datetime.now().strftime("%Y-%m-%d")
    return today, today


def download_data(stock_pool: str = "sp500"):
    """下载最新数据"""
    import sys

    # 1. 下载美股数据
    if not run_command(
        [sys.executable, str(SCRIPTS_DIR / "data" / "download_us_data_to_date.py"),
         "--pool", stock_pool],
        "Downloading latest US stock data"
    ):
        print("Warning: Stock data download had issues, continuing...")

    # 2. 增量更新宏观数据
    if not run_command(
        [sys.executable, str(SCRIPTS_DIR / "data" / "download_macro_data_to_date.py")],
        "Updating macro data to latest date"
    ):
        print("Warning: Macro data download had issues, continuing...")

    # 3. 处理宏观数据
    if not run_command(
        [sys.executable, str(SCRIPTS_DIR / "data" / "process_macro_data.py")],
        "Processing macro data into features"
    ):
        print("Warning: Macro data processing had issues, continuing...")


def init_qlib_for_talib():
    """初始化 Qlib（支持 TA-Lib）"""
    import qlib
    from qlib.constant import REG_US
    from utils.talib_ops import TALIB_OPS

    qlib_data_path = PROJECT_ROOT / "my_data" / "qlib_us"

    qlib.init(
        provider_uri=str(qlib_data_path),
        region=REG_US,
        custom_ops=TALIB_OPS,
        kernels=1,  # 单线程避免 TA-Lib 冲突
    )
    print(f"Qlib initialized with data from: {qlib_data_path}")


def create_dataset_for_trading(handler_name: str, stock_pool: str, test_start: str, test_end: str, nday: int = 5, verbose: bool = True):
    """创建用于交易的数据集

    Parameters
    ----------
    handler_name : str
        Handler 名称
    stock_pool : str
        股票池名称
    test_start : str
        测试开始日期
    test_end : str
        测试结束日期
    nday : int
        N日前向预测
    verbose : bool
        是否打印详细信息
    """
    from qlib.data.dataset import DatasetH
    from models.common.handlers import get_handler_class
    from data.stock_pools import STOCK_POOLS

    if verbose:
        print(f"\nCreating dataset...")
        print(f"  Handler: {handler_name}")
        print(f"  Stock pool: {stock_pool} ({len(STOCK_POOLS[stock_pool])} stocks)")
        print(f"  Test period: {test_start} to {test_end}")
        print(f"  N-day forward: {nday}")

    HandlerClass = get_handler_class(handler_name)
    symbols = STOCK_POOLS[stock_pool]

    handler = HandlerClass(
        instruments=symbols,
        start_time="2024-01-01",
        end_time=test_end,
        fit_start_time="2024-01-01",
        fit_end_time="2025-12-31",
        volatility_window=nday,
        learn_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
        infer_processors=[
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ],
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2024-01-01", "2025-06-30"),
            "valid": ("2025-07-01", "2025-12-31"),
            "test": (test_start, test_end),
        },
    )

    return dataset


def collect_trading_details_from_positions(positions: dict, report_df: pd.DataFrame) -> list:
    """从 positions 字典中收集交易详情

    Parameters
    ----------
    positions : dict
        日期 -> Position 对象的字典
    report_df : pd.DataFrame
        包含每日收益的报告 DataFrame

    Returns
    -------
    list
        交易详情列表
    """
    sorted_dates = sorted(positions.keys())
    prev_holdings = {}
    all_trading_details = []

    for date in sorted_dates:
        pos = positions[date]

        current_holdings = {}
        stock_values = {}
        if hasattr(pos, 'get_stock_list'):
            for stock in pos.get_stock_list():
                amount = pos.get_stock_amount(stock)
                if amount > 0:
                    current_holdings[stock] = amount
                    try:
                        price = pos.get_stock_price(stock)
                        stock_values[stock] = amount * price if price else 0
                    except:
                        stock_values[stock] = 0

        total_value = pos.calculate_value() if hasattr(pos, 'calculate_value') else 0
        cash = pos.get_cash() if hasattr(pos, 'get_cash') else 0

        buys = []
        sells = []

        for stock, amount in current_holdings.items():
            prev_amount = prev_holdings.get(stock, 0)
            diff = amount - prev_amount
            if diff >= 0.5:
                buys.append((stock, diff, amount))

        for stock, prev_amount in prev_holdings.items():
            current_amount = current_holdings.get(stock, 0)
            diff = prev_amount - current_amount
            if diff >= 0.5:
                sells.append((stock, diff, current_amount))

        date_str = str(date)[:10]
        daily_return = report_df.loc[date, 'return'] if date in report_df.index else 0

        day_detail = {
            'date': date_str,
            'total_value': total_value,
            'cash': cash,
            'return': daily_return,
            'sells': [{'stock': s[0], 'amount': s[1], 'remaining': s[2]} for s in sells],
            'buys': [{'stock': b[0], 'amount': b[1], 'total': b[2]} for b in buys],
            'holdings': [{'stock': s, 'amount': a, 'value': stock_values.get(s, 0)}
                         for s, a in sorted(current_holdings.items())],
        }
        all_trading_details.append(day_detail)

        prev_holdings = current_holdings.copy()

    return all_trading_details


def print_trading_details(trading_details: list, show_last_n: int = None):
    """打印交易详情

    Parameters
    ----------
    trading_details : list
        交易详情列表
    show_last_n : int, optional
        只显示最后 N 天，默认显示全部
    """
    if show_last_n:
        display_details = trading_details[-show_last_n:]
    else:
        display_details = trading_details

    for day in display_details:
        print(f"\n{'-'*60}")
        print(f"Date: {day['date']}  |  Total Value: ${day['total_value']:,.2f}  |  Cash: ${day['cash']:,.2f}  |  Daily Return: {day['return']:.2%}")

        if day['sells']:
            sells_str = ', '.join([f"{s['stock'].upper()}({s['amount']:.0f})" for s in day['sells']])
            print(f"  SELL: {sells_str}")

        if day['buys']:
            buys_str = ', '.join([f"{b['stock'].upper()}({b['amount']:.0f})" for b in day['buys']])
            print(f"  BUY:  {buys_str}")

        if day['holdings']:
            holdings_str = ', '.join([h['stock'].upper() for h in day['holdings']])
            print(f"  HOLD: {holdings_str}")


def generate_detailed_trade_records(positions: dict, output_dir: Path, freq: str):
    """生成详细的交易记录 CSV"""
    trade_records = []
    position_records = []

    sorted_dates = sorted(positions.keys())
    prev_holdings = {}

    for date in sorted_dates:
        pos = positions[date]

        current_holdings = {}
        if hasattr(pos, 'get_stock_list'):
            for stock in pos.get_stock_list():
                amount = pos.get_stock_amount(stock)
                if amount > 0:
                    current_holdings[stock] = amount

                    try:
                        price = pos.get_stock_price(stock)
                        value = amount * price if price else 0
                    except:
                        price = 0
                        value = 0

                    position_records.append({
                        'date': date,
                        'stock': stock,
                        'shares': amount,
                        'price': price,
                        'value': value,
                    })

        for stock, amount in current_holdings.items():
            prev_amount = prev_holdings.get(stock, 0)
            if amount > prev_amount:
                trade_records.append({
                    'date': date,
                    'stock': stock,
                    'action': 'BUY',
                    'shares': amount - prev_amount,
                    'position_after': amount,
                })

        for stock, prev_amount in prev_holdings.items():
            current_amount = current_holdings.get(stock, 0)
            if current_amount < prev_amount:
                trade_records.append({
                    'date': date,
                    'stock': stock,
                    'action': 'SELL',
                    'shares': prev_amount - current_amount,
                    'position_after': current_amount,
                })

        prev_holdings = current_holdings.copy()

    if trade_records:
        trade_df = pd.DataFrame(trade_records)
        trade_df = trade_df.sort_values(['date', 'action', 'stock'])
        trade_path = output_dir / f"trade_records_{freq}.csv"
        trade_df.to_csv(trade_path, index=False)
        print(f"Trade records saved to: {trade_path}")
        print(f"  Total trades: {len(trade_df)}")
        print(f"  Buy orders: {len(trade_df[trade_df['action'] == 'BUY'])}")
        print(f"  Sell orders: {len(trade_df[trade_df['action'] == 'SELL'])}")

    if position_records:
        pos_df = pd.DataFrame(position_records)
        pos_path = output_dir / f"position_records_{freq}.csv"
        pos_df.to_csv(pos_path, index=False)
        print(f"Position records saved to: {pos_path}")


# ============================================================================
# Model Loading
# ============================================================================

def load_model_meta(model_path: Path) -> dict:
    """Load model metadata"""
    meta_path = model_path.with_suffix('.meta.pkl')
    if meta_path.exists():
        with open(meta_path, 'rb') as f:
            return pickle.load(f)

    # Try replacing _best suffix
    stem = model_path.stem
    if stem.endswith('_best'):
        alt_meta_path = model_path.parent / (stem[:-5] + '.meta.pkl')
        if alt_meta_path.exists():
            with open(alt_meta_path, 'rb') as f:
                return pickle.load(f)

    return {}


def load_ae_mlp_model(model_path: Path):
    """加载 AE-MLP 模型"""
    from models.deep.ae_mlp_model import AEMLP

    print(f"    Loading AE-MLP model from: {model_path}")
    model = AEMLP.load(str(model_path))
    print(f"    AE-MLP loaded: {model.num_columns} features")
    return model


def load_catboost_model(model_path: Path):
    """加载 CatBoost 模型"""
    from catboost import CatBoostRegressor

    print(f"    Loading CatBoost model from: {model_path}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    feature_count = model.feature_names_ if model.feature_names_ else "N/A"
    if isinstance(feature_count, list):
        feature_count = len(feature_count)
    print(f"    CatBoost loaded: {feature_count} features")
    return model


# ============================================================================
# Prediction
# ============================================================================

def predict_with_ae_mlp(model, dataset) -> pd.Series:
    """Generate predictions with AE-MLP model"""
    pred = model.predict(dataset, segment="test")
    pred.name = 'score'
    return pred


def predict_with_catboost(model, dataset) -> pd.Series:
    """Generate predictions with CatBoost model"""
    from qlib.data.dataset.handler import DataHandlerLP

    test_data = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_L)
    test_data = test_data.fillna(0).replace([np.inf, -np.inf], 0)

    pred_values = model.predict(test_data.values)
    pred = pd.Series(pred_values, index=test_data.index, name='score')
    return pred


# ============================================================================
# Ensemble Utilities
# ============================================================================

def zscore_by_day(x: pd.Series) -> pd.Series:
    """Z-score normalize within each trading day"""
    mean = x.groupby(level='datetime').transform('mean')
    std = x.groupby(level='datetime').transform('std')
    return (x - mean) / (std + 1e-8)


def ensemble_predictions_multi(pred_dict: dict, method: str = 'zscore_weighted',
                                weights: dict = None) -> pd.Series:
    """
    Ensemble multiple model predictions.

    Parameters
    ----------
    pred_dict : dict
        {model_name: pd.Series}
    method : str
        'mean', 'weighted', 'rank_mean', 'zscore_mean', 'zscore_weighted'
    weights : dict, optional
        {model_name: weight}
    """
    names = list(pred_dict.keys())

    # Find common index
    common_idx = pred_dict[names[0]].index
    for name in names[1:]:
        common_idx = common_idx.intersection(pred_dict[name].index)

    preds = {name: pred_dict[name].loc[common_idx] for name in names}

    if method == 'mean':
        ensemble_pred = sum(preds.values()) / len(preds)
    elif method == 'weighted':
        if weights is None:
            weights = {n: 1.0 / len(names) for n in names}
        total = sum(weights.values())
        ensemble_pred = sum(preds[n] * weights[n] for n in names) / total
    elif method == 'rank_mean':
        ranks = {n: preds[n].groupby(level='datetime').rank(pct=True) for n in names}
        ensemble_pred = sum(ranks.values()) / len(ranks)
    elif method == 'zscore_mean':
        zscores = {n: zscore_by_day(preds[n]) for n in names}
        ensemble_pred = sum(zscores.values()) / len(zscores)
    elif method == 'zscore_weighted':
        if weights is None:
            weights = {n: 1.0 / len(names) for n in names}
        total = sum(weights.values())
        zscores = {n: zscore_by_day(preds[n]) for n in names}
        ensemble_pred = sum(zscores[n] * weights[n] for n in names) / total
    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    ensemble_pred.name = 'score'
    return ensemble_pred


def calculate_pairwise_correlations(preds: dict) -> pd.DataFrame:
    """Calculate pairwise correlations between model predictions"""
    model_names = list(preds.keys())
    n_models = len(model_names)

    common_idx = preds[model_names[0]].index
    for name in model_names[1:]:
        common_idx = common_idx.intersection(preds[name].index)

    corr_matrix = np.zeros((n_models, n_models))
    for i, name_i in enumerate(model_names):
        for j, name_j in enumerate(model_names):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                p_i = preds[name_i].loc[common_idx]
                p_j = preds[name_j].loc[common_idx]
                corr_matrix[i, j] = p_i.corr(p_j)

    return pd.DataFrame(corr_matrix, index=model_names, columns=model_names)


# ============================================================================
# Date Detection / Validation
# ============================================================================

def detect_and_validate_dates(args) -> None:
    """
    自动检测数据日期范围并验证/调整用户提供的日期参数。
    直接修改 args 对象：设置 args.test_start, args.backtest_start, args.test_end
    """
    print(f"\n{'='*60}")
    print("[STEP] Detecting data date range")
    print(f"{'='*60}")
    latest_data_str = get_latest_data_date()
    latest_calendar_str, usable_calendar_str = get_latest_calendar_dates()
    latest_data = datetime.strptime(latest_data_str, "%Y-%m-%d")
    usable_calendar = datetime.strptime(usable_calendar_str, "%Y-%m-%d")

    print(f"Latest available data date: {latest_data_str}")
    print(f"Latest Qlib calendar date: {latest_calendar_str}")
    print(f"Usable calendar date (for backtest end): {usable_calendar_str}")

    max_backtest_date = min(latest_data, usable_calendar)
    max_backtest_str = max_backtest_date.strftime("%Y-%m-%d")

    backtest_start_date = datetime.strptime(args.backtest_start, "%Y-%m-%d")

    if backtest_start_date > max_backtest_date:
        print(f"\nWARNING: Backtest start date {args.backtest_start} exceeds usable range!")
        print(f"         Automatically adjusting to: {max_backtest_str}")
        backtest_start_date = max_backtest_date
        args.backtest_start = max_backtest_str

    data_start_date = backtest_start_date - timedelta(days=args.feature_lookback)
    args.test_start = data_start_date.strftime("%Y-%m-%d")

    if args.test_end is None:
        args.test_end = max_backtest_str

    test_end_date = datetime.strptime(args.test_end, "%Y-%m-%d")
    if test_end_date > max_backtest_date:
        print(f"\nWARNING: Test end date {args.test_end} exceeds usable range!")
        print(f"         Automatically adjusting to: {max_backtest_str}")
        args.test_end = max_backtest_str

    print(f"\nFinal date settings:")
    print(f"  Data start (for features): {args.test_start}")
    print(f"  Backtest start: {args.backtest_start}")
    print(f"  Backtest end: {args.test_end}")


# ============================================================================
# Email Sending
# ============================================================================

def send_trading_email(args, trading_details: list, model_info: str = None) -> None:
    """
    根据 args.send_email, args.email_days, args.email_to 发送交易报告邮件。
    """
    if not args.send_email or not trading_details:
        return

    print(f"\n{'='*60}")
    print("[STEP] Sending email report")
    print(f"{'='*60}")

    recent_details = trading_details[-args.email_days:] if len(trading_details) > args.email_days else trading_details

    if not recent_details:
        print("No trading details to send.")
        return

    text_body = format_trading_details_text(recent_details, model_info or "")
    html_body = format_trading_details_html(recent_details, model_info or "")

    start_date = recent_details[0]['date']
    end_date = recent_details[-1]['date']
    prefix = f"{model_info} " if model_info else ""
    subject = f"{prefix}Trading Report: {start_date} to {end_date}"

    send_email(
        subject=subject,
        body=text_body,
        to_email=args.email_to,
        html_body=html_body
    )


# ============================================================================
# Ensemble Live Prediction
# ============================================================================

def run_ensemble_live_prediction(
    pred_ensemble: pd.Series,
    stock_pool: str,
    topk: int = 10,
    account: float = 8000,
    version_label: str = "Ensemble",
    file_prefix: str = "ensemble",
) -> pd.DataFrame:
    """
    生成 ensemble 实盘交易预测（不运行回测）。

    Parameters
    ----------
    pred_ensemble : pd.Series
        集成后的预测信号
    stock_pool : str
        股票池名称
    topk : int
        持仓数量
    account : float
        初始资金
    version_label : str
        标题中显示的版本标签
    file_prefix : str
        输出文件名前缀
    """
    print(f"\n{'='*70}")
    print(f"LIVE TRADING PREDICTIONS ({version_label})")
    print(f"{'='*70}")
    print(f"Account: ${account:,.2f}")
    print(f"Top-K stocks: {topk}")

    pred_df = pred_ensemble.to_frame("score")
    pred_df = pred_df.reset_index()
    pred_df['instrument'] = pred_df['instrument'].str.lower()
    pred_df = pred_df.set_index(['datetime', 'instrument'])

    pred_dates = pred_df.index.get_level_values(0).unique().sort_values()
    print(f"\nPredictions available for: {pred_dates.min().strftime('%Y-%m-%d')} to {pred_dates.max().strftime('%Y-%m-%d')}")
    print(f"Total dates: {len(pred_dates)}, Total predictions: {len(pred_df)}")

    latest_date = pred_dates[-1]
    latest_preds = pred_df.loc[latest_date].sort_values("score", ascending=False)

    print(f"\n{'='*70}")
    print(f"TRADING SIGNALS FOR: {latest_date.strftime('%Y-%m-%d')}")
    print(f"{'='*70}")

    print(f"\nTOP {topk} STOCKS TO BUY (highest predicted return):")
    print("-" * 60)
    print(f"{'Rank':<6} {'Symbol':<10} {'Score':>12} {'Est. Allocation':>18}")
    print("-" * 60)

    allocation_per_stock = account * 0.95 / topk
    top_stocks = latest_preds.head(topk)

    for i, (stock, row) in enumerate(top_stocks.iterrows(), 1):
        print(f"{i:<6} {stock.upper():<10} {row['score']:>12.4f} ${allocation_per_stock:>15,.2f}")

    print(f"\nTOP {topk} STOCKS TO AVOID (lowest predicted return):")
    print("-" * 60)
    bottom_stocks = latest_preds.tail(topk).iloc[::-1]
    for i, (stock, row) in enumerate(bottom_stocks.iterrows(), 1):
        print(f"{i:<6} {stock.upper():<10} {row['score']:>12.4f}")

    # 保存预测结果
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_output = latest_preds.copy()
    pred_output.index = pred_output.index.str.upper()
    pred_output = pred_output.sort_values("score", ascending=False)
    pred_path = output_dir / f"{file_prefix}_predictions_{latest_date.strftime('%Y%m%d')}.csv"
    pred_output.to_csv(pred_path)
    print(f"\nFull predictions saved to: {pred_path}")

    # 保存交易建议
    recommendations = []
    for i, (stock, row) in enumerate(top_stocks.iterrows(), 1):
        recommendations.append({
            'rank': i,
            'symbol': stock.upper(),
            'action': 'BUY',
            'score': row['score'],
            'suggested_allocation': allocation_per_stock,
        })

    rec_df = pd.DataFrame(recommendations)
    rec_path = output_dir / f"{file_prefix}_recommendations_{latest_date.strftime('%Y%m%d')}.csv"
    rec_df.to_csv(rec_path, index=False)
    print(f"Buy recommendations saved to: {rec_path}")

    print(f"\n{'='*70}")
    print("LIVE PREDICTION COMPLETED")
    print(f"{'='*70}")

    return pred_df


# ============================================================================
# Ensemble Backtest
# ============================================================================

def run_ensemble_backtest(
    pred_ensemble: pd.Series,
    stock_pool: str,
    test_start: str,
    test_end: str,
    account: float = 8000,
    topk: int = 5,
    n_drop: int = 1,
    rebalance_freq: int = 5,
    strategy_type: str = "topk",
    optimizer_params: dict = None,
    version_label: str = "Ensemble",
    file_prefix: str = "ensemble",
    use_signal_shift: bool = True,
    benchmark: str = "SPY",
) -> list:
    """
    运行 ensemble 交易回测，返回交易详情列表。

    Parameters
    ----------
    pred_ensemble : pd.Series
        集成后的预测信号
    stock_pool : str
        股票池名称
    test_start : str
        回测开始日期
    test_end : str
        回测结束日期
    account : float
        初始资金
    topk : int
        持仓数量
    n_drop : int
        每次调仓淘汰数量
    rebalance_freq : int
        调仓频率（天）
    strategy_type : str
        策略类型
    optimizer_params : dict
        优化器参数
    version_label : str
        标题中显示的版本标签
    file_prefix : str
        输出文件名前缀
    use_signal_shift : bool
        是否使用 signal shift（回测从第二个预测日开始）
    benchmark : str
        基准（"SPY" 或其他）
    """
    from qlib.backtest import backtest as qlib_backtest
    from qlib.contrib.evaluate import risk_analysis
    from data.stock_pools import STOCK_POOLS
    from utils.strategy import get_strategy_config

    strategy_label = strategy_type.upper() if strategy_type in ('mvo', 'rp', 'gmv', 'inv') else strategy_type
    print(f"\n{'='*70}")
    print(f"TRADING BACKTEST - {strategy_label} ({version_label})")
    print(f"{'='*70}")
    print(f"Period: {test_start} to {test_end}")
    print(f"Initial Account: ${account:,.2f}")
    print(f"Strategy: {strategy_type}")
    print(f"Rebalance Frequency: every {rebalance_freq} day(s)")
    print(f"Top-K stocks: {topk}")
    if strategy_type not in ('mvo', 'rp', 'gmv', 'inv'):
        print(f"N-drop: {n_drop}")
    if optimizer_params:
        if strategy_type == "mvo":
            print(f"MVO lamb: {optimizer_params.get('lamb', 2.0)}")
        print(f"Max weight: {optimizer_params.get('max_weight', 0):.0%}" if optimizer_params.get('max_weight', 0) > 0 else "Max weight: unlimited")

    # 转换为 DataFrame
    pred_df = pred_ensemble.to_frame("score")
    pred_df = pred_df.reset_index()
    pred_df['instrument'] = pred_df['instrument'].str.lower()
    pred_df = pred_df.set_index(['datetime', 'instrument'])

    pred_dates = pred_df.index.get_level_values(0).unique().sort_values()
    print(f"\nPredictions shape: {len(pred_df)}")
    print(f"Date range: {pred_dates.min()} to {pred_dates.max()}")
    print(f"Unique dates: {len(pred_dates)}")

    # 显示最新日期的推荐股票
    latest_pred_date = pred_dates[-1]
    print(f"\n{'='*70}")
    print(f"TODAY'S RECOMMENDATIONS (based on {latest_pred_date.strftime('%Y-%m-%d')} predictions)")
    print(f"{'='*70}")

    latest_preds = pred_df.loc[latest_pred_date].sort_values("score", ascending=False)
    print(f"\nTop {topk} stocks to BUY:")
    print("-" * 50)
    for i, (stock, row) in enumerate(latest_preds.head(topk).iterrows(), 1):
        print(f"  {i:2d}. {stock.upper():<8s}  Score: {row['score']:>8.4f}")

    # 调整回测日期
    if use_signal_shift and len(pred_dates) > 1:
        # qlib signal_strategy uses shift=1: trade day T uses T-1 prediction
        actual_test_end = pred_dates[-1].strftime("%Y-%m-%d")
        actual_test_start = pred_dates[1].strftime("%Y-%m-%d")

        if pd.Timestamp(test_start) > pred_dates[1]:
            actual_test_start = test_start

        if actual_test_start != test_start or actual_test_end != test_end:
            print(f"\n[NOTE] Adjusting backtest period:")
            print(f"       Original: {test_start} to {test_end}")
            print(f"       Adjusted: {actual_test_start} to {actual_test_end}")
            test_start = actual_test_start
            test_end = actual_test_end
    elif not use_signal_shift and len(pred_dates) > 0:
        actual_test_end = pred_dates[-1].strftime("%Y-%m-%d")
        actual_test_start = pred_dates[0].strftime("%Y-%m-%d")

        if pd.Timestamp(test_start) > pred_dates[0]:
            actual_test_start = test_start

        if actual_test_start != test_start or actual_test_end != test_end:
            print(f"\n[NOTE] Adjusting backtest period:")
            print(f"       Original: {test_start} to {test_end}")
            print(f"       Adjusted: {actual_test_start} to {actual_test_end}")
            test_start = actual_test_start
            test_end = actual_test_end

    # 配置策略
    strategy_config = get_strategy_config(
        pred_df, topk, n_drop, rebalance_freq=rebalance_freq,
        strategy_type=strategy_type,
        optimizer_params=optimizer_params,
    )

    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }

    pool_symbols = STOCK_POOLS[stock_pool]
    pool_symbols_lower = [s.lower() for s in pool_symbols]

    backtest_config = {
        "start_time": test_start,
        "end_time": test_end,
        "account": account,
        "benchmark": benchmark,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": None,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0005,
            "min_cost": 1,
            "trade_unit": None,
            "codes": pool_symbols_lower,
        },
    }

    # 运行回测
    print(f"\n[*] Running backtest...")
    portfolio_metric_dict, indicator_dict = qlib_backtest(
        executor=executor_config,
        strategy=strategy_config,
        **backtest_config
    )

    print("    Backtest completed!")

    # 分析结果
    for freq, (report_df, positions) in portfolio_metric_dict.items():
        print(f"\n{'='*70}")
        print(f"TRADING RESULTS ({freq})")
        print(f"{'='*70}")

        total_return = (report_df["return"] + 1).prod() - 1
        final_value = account * (1 + total_return)

        print(f"\nPerformance Summary:")
        print(f"  Initial Capital:  ${account:>12,.2f}")
        print(f"  Final Value:      ${final_value:>12,.2f}")
        print(f"  Total Return:     {total_return:>12.2%}")
        print(f"  Total P&L:        ${final_value - account:>12,.2f}")

        if "bench" in report_df.columns and not report_df["bench"].isna().all():
            bench_return = (report_df["bench"] + 1).prod() - 1
            excess_return = total_return - bench_return
            print(f"  Benchmark Return: {bench_return:>12.2%}")
            print(f"  Excess Return:    {excess_return:>12.2%}")

        analysis = risk_analysis(report_df["return"], freq=freq)
        if analysis is not None and not analysis.empty:
            print(f"\nRisk Metrics:")
            for metric, value in analysis.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric:<20s}: {value:>10.4f}")

        print(f"\nDaily Statistics:")
        print(f"  Trading Days:     {len(report_df):>12d}")
        print(f"  Mean Daily Return:{report_df['return'].mean():>12.4%}")
        print(f"  Std Daily Return: {report_df['return'].std():>12.4%}")
        print(f"  Best Day:         {report_df['return'].max():>12.4%}")
        print(f"  Worst Day:        {report_df['return'].min():>12.4%}")

        # 输出每日详细交易信息
        print(f"\n{'='*70}")
        print("DAILY TRADING DETAILS (Last 10 days)")
        print(f"{'='*70}")

        all_trading_details = collect_trading_details_from_positions(positions, report_df)
        print_trading_details(all_trading_details, show_last_n=10)

        # 保存报告
        output_dir = PROJECT_ROOT / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / f"{file_prefix}_daily_trading_report_{freq}.csv"
        report_df.to_csv(report_path)
        print(f"\n\nReport saved to: {report_path}")

        return all_trading_details

    return []


# ============================================================================
# Common Argparse
# ============================================================================

def add_common_trading_args(parser: argparse.ArgumentParser) -> None:
    """添加所有 trading 脚本共享的参数"""
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download step')
    parser.add_argument('--predict-only', action='store_true',
                        help='Only generate predictions, skip backtest')
    parser.add_argument('--stock-pool', type=str, default='sp500',
                        choices=['test', 'tech', 'sp100', 'sp500'],
                        help='Stock pool')
    parser.add_argument('--account', type=float, default=8000,
                        help='Initial account balance (default: 8000)')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of stocks to hold (default: 10)')
    parser.add_argument('--n-drop', type=int, default=2,
                        help='Number of stocks to drop each rebalance (default: 2)')
    parser.add_argument('--rebalance-freq', type=int, default=5,
                        help='Rebalance frequency in days (default: 5)')
    parser.add_argument('--backtest-start', type=str, default='2026-02-01',
                        help='Backtest start date')
    parser.add_argument('--test-end', type=str, default=None,
                        help='Backtest end date (default: latest available data date)')
    parser.add_argument('--nday', type=int, default=5,
                        help='N-day forward prediction (default: 5)')
    parser.add_argument('--feature-lookback', type=int, default=5,
                        help='Days before backtest-start for feature calculation (default: 5)')
    parser.add_argument('--send-email', action='store_true',
                        help='Send trading report via email')
    parser.add_argument('--email-to', type=str, default='kownse@gmail.com',
                        help='Email recipient (default: kownse@gmail.com)')
    parser.add_argument('--email-days', type=int, default=5,
                        help='Number of recent days to include in email (default: 5)')
