"""
LLM-based fundamental analysis filter for stock screening.

Reviews selected stocks using Claude API to identify fundamental risks
(governance issues, earnings problems, AI disruption, etc.) that
technical models cannot detect.

Results are permanently cached to outputs/llm_stock_analysis/{SYMBOL}.json.
"""

import json
from datetime import datetime
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "outputs" / "llm_stock_analysis"

SYSTEM_PROMPT = """你是一位专业的美股基本面分析师。用户会给你一只股票代码，请你根据你的知识进行全面评估。

请分析以下几个方面：
1. 该公司是否处于被 AI 替代/颠覆的行业？（如传统软件 SaaS、IT外包等）
2. 过去 1 年股价是否大幅下跌？如果是，主要原因是什么？
3. 公司是否存在治理问题（SEC调查、财务造假、管理层丑闻等）？
4. 如果最近上涨，上涨原因是什么？是否可持续？
5. 公司增长前景如何？

请以 JSON 格式回答，字段如下：
{
  "verdict": "pass" 或 "warning" 或 "fail",
  "risk_level": 1-5 的整数（1=极低风险, 5=极高风险）,
  "ai_disruption": true/false,
  "decline_reason": "如果近期大跌，说明原因；否则 null",
  "recovery_potential": "high/medium/low/null",
  "growth_sustainability": "strong/moderate/weak/uncertain",
  "summary": "一段简要分析（2-3句话）",
  "recommendation": "buy/hold/avoid"
}

评判标准：
- fail: 有严重治理问题（SEC调查、造假）、被AI彻底颠覆的行业、或基本面严重恶化
- warning: 有一定风险但不致命，需要关注
- pass: 基本面健康，没有重大红旗

只输出 JSON，不要输出其他内容。"""


def _load_cache(symbol: str) -> dict | None:
    """Load cached analysis for a symbol, or None if not cached."""
    cache_file = CACHE_DIR / f"{symbol.upper()}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cache(symbol: str, result: dict):
    """Save analysis result to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{symbol.upper()}.json"
    with open(cache_file, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def analyze_stock(symbol: str, model: str = "claude-sonnet-4-6") -> dict:
    """
    Analyze a single stock's fundamentals using Claude API.

    Checks cache first; if not cached, calls Claude API and saves result.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., "SMCI").
    model : str
        Claude model to use.

    Returns
    -------
    dict
        Analysis result with verdict, risk_level, summary, etc.
    """
    symbol = symbol.upper()

    # Check cache
    cached = _load_cache(symbol)
    if cached is not None:
        print(f"    [LLM] {symbol} → CACHE HIT | verdict={cached.get('verdict')} risk={cached.get('risk_level')} | {cached.get('summary', '')[:80]}")
        return cached

    # Call Claude API
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"请分析股票: {symbol}"
            }],
        )

        response_text = response.content[0].text.strip()

        # Parse JSON response - handle markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first and last lines (``` markers)
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        result = json.loads(response_text)

        # Add metadata
        result["symbol"] = symbol
        result["model"] = model
        result["analyzed_date"] = datetime.now().strftime("%Y-%m-%d")

        # Validate required fields
        for field in ["verdict", "risk_level", "summary"]:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Normalize verdict
        result["verdict"] = result["verdict"].lower()
        if result["verdict"] not in ("pass", "warning", "fail"):
            result["verdict"] = "warning"

        _save_cache(symbol, result)
        print(f"    [LLM] {symbol} → API CALL  | verdict={result['verdict']} risk={result['risk_level']} | {result['summary'][:80]}")
        return result

    except (anthropic.APIError, json.JSONDecodeError, ValueError, KeyError) as e:
        # Fail-safe: any error defaults to pass
        print(f"    [LLM] Error analyzing {symbol}: {e}")
        return {
            "symbol": symbol,
            "verdict": "pass",
            "risk_level": 0,
            "summary": f"Analysis failed: {e}",
            "recommendation": "hold",
            "model": model,
            "analyzed_date": datetime.now().strftime("%Y-%m-%d"),
            "error": str(e),
        }


def review_selected_stocks(
    symbols: list[str],
    model: str = "claude-sonnet-4-6",
) -> list[dict]:
    """
    Review a list of selected stocks (post top-K selection).

    Parameters
    ----------
    symbols : list[str]
        Stock ticker symbols to review.
    model : str
        Claude model to use.

    Returns
    -------
    list[dict]
        List of analysis results, one per symbol.
    """
    results = []
    cached_count = 0
    api_count = 0

    for symbol in symbols:
        symbol = symbol.upper()
        was_cached = _load_cache(symbol) is not None
        result = analyze_stock(symbol, model=model)
        results.append(result)

        if was_cached:
            cached_count += 1
            status = "cached"
        else:
            api_count += 1
            status = "new"

        verdict = result.get("verdict", "?")
        risk = result.get("risk_level", "?")
        summary = result.get("summary", "")[:60]
        print(f"    [{status:6s}] {symbol:<6s} verdict={verdict:<7s} risk={risk} | {summary}")

    print(f"\n    LLM Review: {len(symbols)} stocks, {cached_count} cached, {api_count} API calls")
    return results


def refresh_stock(symbol: str):
    """Delete cached analysis for a symbol to force re-analysis."""
    cache_file = CACHE_DIR / f"{symbol.upper()}.json"
    if cache_file.exists():
        cache_file.unlink()
        print(f"  Deleted cache for {symbol.upper()}")
    else:
        print(f"  No cache found for {symbol.upper()}")
