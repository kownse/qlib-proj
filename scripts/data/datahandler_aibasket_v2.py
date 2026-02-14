"""Alpha158 + TA-Lib + Macro + AI Basket V2 handler.

Feature-selected version based on CatBoost importance analysis.
Only includes features with importance >= 0.01.
Reduces from ~299 features to ~166 features.
"""
from data.datahandler_macro import Alpha158_Volatility_TALib_Macro


class Alpha158_Volatility_TALib_Macro_AIBasket_V2(Alpha158_Volatility_TALib_Macro):
    """Feature-selected handler: importance >= 0.01 from CatBoost analysis.

    Original: ~299 features (Alpha158 + TA-Lib + Macro + AI Basket)
    V2: ~166 features after importance filtering
    """

    # Alpha158 + TA-Lib features to keep (83 total)
    SELECTED_ALPHA_TALIB = {
        # Alpha158 base features (73)
        "OPEN0", "HIGH0",
        "STD5", "STD10", "STD20", "STD30", "STD60",
        "ROC5", "ROC10", "ROC60",
        "BETA5", "BETA20", "BETA30", "BETA60",
        "MAX5", "MAX10", "MAX20", "MAX30", "MAX60",
        "MIN5", "MIN10", "MIN60",
        "MA5", "MA10", "MA30", "MA60",
        "QTLU5", "QTLU10", "QTLU30", "QTLU60",
        "QTLD5", "QTLD10", "QTLD20", "QTLD30",
        "IMAX10", "IMAX20", "IMAX60",
        "IMIN60",
        "IMXD5", "IMXD10", "IMXD20", "IMXD30", "IMXD60",
        "CORR60",
        "RSQR30", "RSQR60",
        "RESI10", "RESI30", "RESI60",
        "RSV5", "RSV10", "RSV30", "RSV60",
        "VSTD20", "VSTD60",
        "WVMA10", "WVMA20", "WVMA30", "WVMA60",
        "SUMD5", "SUMD30", "SUMD60",
        "SUMN10", "SUMN20", "SUMN30", "SUMN60",
        "SUMP60",
        "VSUMD60",
        "CNTD5", "CNTD20",
        "CNTN10",
        "CNTP30",
        "KSFT",
        # TA-Lib features (10)
        "TALIB_NATR14", "TALIB_ATR14",
        "TALIB_BB_WIDTH", "TALIB_BB_LOWER_DIST",
        "TALIB_MACD_SIGNAL",
        "TALIB_STOCH_K", "TALIB_STOCH_D",
        "TALIB_STDDEV20",
        "TALIB_MOM10",
        "TALIB_EMA20",
    }

    # Macro features to keep (77 total)
    SELECTED_MACRO_FEATURES = [
        # FRED Credit (5)
        "macro_hy_spread", "macro_hy_spread_zscore", "macro_hy_spread_chg5",
        "macro_ig_spread", "macro_credit_risk",
        # Benchmark (6)
        "macro_spy_vol20", "macro_spy_pct_1d", "macro_spy_pct_5d",
        "macro_spy_pct_20d", "macro_spy_ma20_ratio", "macro_qqq_vs_spy",
        # Treasury (8)
        "macro_yield_10y", "macro_yield_2y", "macro_yield_30y",
        "macro_yield_3m10y", "macro_yield_2s10s",
        "macro_yield_10y_chg20",
        "macro_yield_curve_slope", "macro_yield_curve_zscore",
        # VIX (7)
        "macro_vix_level", "macro_vix_zscore20", "macro_vix_regime",
        "macro_vix_pct_5d", "macro_vix_pct_10d",
        "macro_vix_ma5_ratio", "macro_vix_ma20_ratio",
        # VIX Term (3)
        "macro_vix_term_structure",
        "macro_uvxy_pct_5d", "macro_svxy_pct_5d",
        # Gold (4)
        "macro_gld_pct_5d", "macro_gld_pct_20d",
        "macro_gld_ma20_ratio", "macro_gld_vol20",
        # Bond (6)
        "macro_tlt_pct_5d", "macro_tlt_pct_20d",
        "macro_tlt_ma20_ratio", "macro_yield_curve", "macro_yield_curve_chg5",
        "macro_bond_vol20",
        # Dollar (2)
        "macro_uup_ma20_ratio", "macro_uup_strength",
        # Oil (3)
        "macro_uso_pct_5d", "macro_uso_pct_20d", "macro_uso_vol20",
        # Sector ETFs (21)
        "macro_xlk_pct_5d", "macro_xlk_pct_20d", "macro_xlk_vs_spy",
        "macro_xlf_pct_20d",
        "macro_xlv_pct_5d", "macro_xlv_pct_20d", "macro_xlv_vs_spy",
        "macro_xli_pct_5d", "macro_xli_pct_20d", "macro_xli_vs_spy",
        "macro_xlp_pct_5d", "macro_xlp_pct_20d",
        "macro_xly_pct_5d", "macro_xly_pct_20d",
        "macro_xlu_pct_20d",
        "macro_xlre_pct_5d", "macro_xlre_vs_spy",
        "macro_xlb_pct_20d", "macro_xlb_vs_spy",
        "macro_xlc_pct_5d", "macro_xlc_vs_spy",
        # Credit/Risk (5)
        "macro_hyg_pct_5d", "macro_hyg_pct_20d", "macro_hyg_vs_lqd",
        "macro_jnk_vol20", "macro_hyg_tlt_ratio",
        # Global (5)
        "macro_eem_pct_5d", "macro_efa_pct_5d", "macro_efa_vs_spy",
        "macro_ewj_pct_5d", "macro_global_risk",
        # Cross-asset (2)
        "macro_stock_bond_corr", "macro_market_stress",
    ]

    # AI basket features to keep (6 out of 11)
    SELECTED_AI_BASKET_FEATURES = [
        "ai_basket_vol20",
        "ai_basket_dd60",
        "ai_basket_ret_20d",
        "ai_basket_vs_spy",
        "ai_basket_ret_5d",
        "ai_basket_breadth",
    ]

    def get_feature_config(self):
        """Override to only include features with importance >= 0.01."""
        fields, names = super().get_feature_config()
        filtered_fields = []
        filtered_names = []
        for f, n in zip(fields, names):
            if n in self.SELECTED_ALPHA_TALIB:
                filtered_fields.append(f)
                filtered_names.append(n)
        return filtered_fields, filtered_names

    def _get_macro_feature_columns(self):
        """Override to return only important macro features."""
        return self.SELECTED_MACRO_FEATURES

    def _add_ai_basket_to_processed_data(self):
        """Override to only include important AI basket features."""
        try:
            all_cols = self._ai_basket_df.columns.tolist()
            cols = [c for c in self.SELECTED_AI_BASKET_FEATURES if c in all_cols]

            if not cols:
                return

            if hasattr(self, "_learn") and self._learn is not None:
                self._learn = self._merge_macro_to_df(
                    self._learn, cols, source_df=self._ai_basket_df)
                print(f"Added {len(cols)} AI basket features (filtered) to learn data")

            if hasattr(self, "_infer") and self._infer is not None:
                self._infer = self._merge_macro_to_df(
                    self._infer, cols, source_df=self._ai_basket_df)

        except Exception as e:
            print(f"Warning: Error adding AI basket features: {e}")
