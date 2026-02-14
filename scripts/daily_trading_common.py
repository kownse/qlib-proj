#!/usr/bin/env python
"""
Daily Trading Common Utilities

Common functions shared between run_daily_trading.py and run_daily_trading_ensemble.py
"""

import os
import sys
import subprocess
import smtplib
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd

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
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


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
