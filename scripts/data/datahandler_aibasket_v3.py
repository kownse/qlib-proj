"""Alpha158 + TA-Lib + Macro + AI Basket V3 handler.

Feature-selected: top 100 features from V2 CatBoost importance analysis.
"""
from data.datahandler_aibasket_v2 import Alpha158_Volatility_TALib_Macro_AIBasket_V2


class Alpha158_Volatility_TALib_Macro_AIBasket_V3(Alpha158_Volatility_TALib_Macro_AIBasket_V2):
    """Top 100 features from V2 CatBoost importance ranking.

    V2: ~166 features → V3: 100 features (top 100 by importance)
    Breakdown: 34 Alpha158 + 7 TA-Lib + 54 Macro + 5 AI Basket
    """

    SELECTED_ALPHA_TALIB = {
        # Alpha158 base features (34)
        "STD10", "STD20", "STD30", "STD60",
        "ROC60",
        "BETA5", "BETA30", "BETA60",
        "MAX5", "MAX10", "MAX60",
        "MIN5", "MIN60",
        "MA5", "MA30", "MA60",
        "QTLU5", "QTLU10", "QTLU60",
        "QTLD20", "QTLD30",
        "IMAX20", "IMAX60",
        "IMIN60",
        "IMXD30", "IMXD60",
        "CORR60",
        "RSQR60",
        "RESI60",
        "RSV60",
        "VSTD60",
        "WVMA60",
        "SUMD60",
        "SUMP60",
        # TA-Lib features (7)
        "TALIB_ATR14", "TALIB_NATR14",
        "TALIB_BB_LOWER_DIST", "TALIB_BB_WIDTH",
        "TALIB_MACD_SIGNAL",
        "TALIB_STOCH_D",
        "TALIB_STDDEV20",
    }

    SELECTED_MACRO_FEATURES = [
        # FRED Credit (4)
        "macro_hy_spread", "macro_hy_spread_zscore",
        "macro_ig_spread", "macro_credit_risk",
        # Benchmark (3)
        "macro_spy_vol20", "macro_spy_pct_20d", "macro_spy_ma20_ratio",
        # Treasury (8)
        "macro_yield_10y", "macro_yield_2y", "macro_yield_30y",
        "macro_yield_3m10y", "macro_yield_2s10s",
        "macro_yield_10y_chg20",
        "macro_yield_curve_slope", "macro_yield_curve_zscore",
        # VIX (4)
        "macro_vix_level", "macro_vix_zscore20",
        "macro_vix_regime", "macro_vix_ma20_ratio",
        # VIX Term (2)
        "macro_uvxy_pct_5d", "macro_svxy_pct_5d",
        # Gold (3)
        "macro_gld_pct_20d", "macro_gld_ma20_ratio", "macro_gld_vol20",
        # Bond (4)
        "macro_tlt_pct_20d", "macro_tlt_ma20_ratio",
        "macro_yield_curve", "macro_bond_vol20",
        # Dollar (2)
        "macro_uup_ma20_ratio", "macro_uup_strength",
        # Oil (3)
        "macro_uso_pct_5d", "macro_uso_pct_20d", "macro_uso_vol20",
        # Sector ETFs (13)
        "macro_xlk_pct_20d",
        "macro_xlf_pct_20d",
        "macro_xlv_pct_5d", "macro_xlv_pct_20d",
        "macro_xli_pct_5d", "macro_xli_pct_20d",
        "macro_xlp_pct_20d",
        "macro_xly_pct_20d",
        "macro_xlu_pct_20d",
        "macro_xlre_pct_5d",
        "macro_xlb_pct_20d",
        "macro_xlc_pct_5d", "macro_xlc_vs_spy",
        # Credit/Risk (4)
        "macro_hyg_pct_20d", "macro_hyg_vs_lqd",
        "macro_jnk_vol20", "macro_hyg_tlt_ratio",
        # Global (2)
        "macro_eem_pct_5d", "macro_ewj_pct_5d",
        # Cross-asset (2)
        "macro_stock_bond_corr", "macro_global_risk",
    ]

    SELECTED_AI_BASKET_FEATURES = [
        "ai_basket_vol20",
        "ai_basket_dd60",
        "ai_basket_vs_spy",
        "ai_basket_ret_20d",
        "ai_basket_ret_5d",
    ]
