"""Supported datasets and legacy-type translation helpers."""

from __future__ import annotations
import warnings
from typing import Optional
import pandas as pd

__all__ = ["list_supported_datasets"]


# %% Supported Fama-French datasets
# List of dicts of the supported Fama-French datasets, including their
# names and frequencies (daily, weekly, monthly). Each dataset type is
# associated with a specific Fama-French model (e.g., 3 factors,
# 5 factors). Annotated with the domain "Fama-French".
_FF_DATASETS = [
    {"type": "factors_ff_3_monthly", "dataset_name": "Fama/French 3 Factors", "file_url": "ftp/F-F_Research_Data_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_3_weekly", "dataset_name": "Fama/French 3 Factors [Weekly]", "file_url": "ftp/F-F_Research_Data_Factors_weekly_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_3_daily", "dataset_name": "Fama/French 3 Factors [Daily]", "file_url": "ftp/F-F_Research_Data_Factors_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_5_2x3_monthly", "dataset_name": "Fama/French 5 Factors (2x3)", "file_url": "ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_5_2x3_daily", "dataset_name": "Fama/French 5 Factors (2x3) [Daily]", "file_url": "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_size_monthly", "dataset_name": "Portfolios Formed on Size", "file_url": "ftp/Portfolios_Formed_on_ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_size_exdividends_monthly", "dataset_name": "Portfolios Formed on Size [ex.Dividends]", "file_url": "ftp/Portfolios_Formed_on_ME_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_size_daily", "dataset_name": "Portfolios Formed on Size [Daily]", "file_url": "ftp/Portfolios_Formed_on_ME_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_bm_monthly", "dataset_name": "Portfolios Formed on Book-to-Market", "file_url": "ftp/Portfolios_Formed_on_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_bm_exdividends_monthly", "dataset_name": "Portfolios Formed on Book-to-Market [ex. Dividends]", "file_url": "ftp/Portfolios_Formed_on_BE-ME_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_bm_daily", "dataset_name": "Portfolios Formed on Book-to-Market [Daily]", "file_url": "ftp/Portfolios_Formed_on_BE-ME_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_op_monthly", "dataset_name": "Portfolios Formed on Operating Profitability", "file_url": "ftp/Portfolios_Formed_on_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_op_exdividends_monthly", "dataset_name": "Portfolios Formed on Operating Profitability [ex. Dividends]", "file_url": "ftp/Portfolios_Formed_on_OP_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_op_daily", "dataset_name": "Portfolios Formed on Operating Profitability [Daily]", "file_url": "ftp/Portfolios_Formed_on_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_inv_monthly", "dataset_name": "Portfolios Formed on Investment", "file_url": "ftp/Portfolios_Formed_on_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_inv_exdividends_monthly", "dataset_name": "Portfolios Formed on Investment [ex. Dividends]", "file_url": "ftp/Portfolios_Formed_on_INV_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_inv_daily", "dataset_name": "Portfolios Formed on Investment [Daily]", "file_url": "ftp/Portfolios_Formed_on_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_bm_2_x_3_monthly", "dataset_name": "6 Portfolios Formed on Size and Book-to-Market (2 x 3)", "file_url": "ftp/6_Portfolios_2x3_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_bm_2_x_3_exdividends_monthly", "dataset_name": "6 Portfolios Formed on Size and Book-to-Market (2 x 3) [ex. Dividends]", "file_url": "ftp/6_Portfolios_2x3_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_bm_2_x_3_weekly", "dataset_name": "6 Portfolios Formed on Size and Book-to-Market (2 x 3) [Weekly]", "file_url": "ftp/6_Portfolios_2x3_weekly_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_bm_2_x_3_daily", "dataset_name": "6 Portfolios Formed on Size and Book-to-Market (2 x 3) [Daily]", "file_url": "ftp/6_Portfolios_2x3_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_bm_5_x_5_monthly", "dataset_name": "25 Portfolios Formed on Size and Book-to-Market (5 x 5)", "file_url": "ftp/25_Portfolios_5x5_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_bm_5_x_5_exdividends_monthly", "dataset_name": "25 Portfolios Formed on Size and Book-to-Market (5 x 5) [ex. Dividends]", "file_url": "ftp/25_Portfolios_5x5_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_bm_5_x_5_daily", "dataset_name": "25 Portfolios Formed on Size and Book-to-Market (5 x 5) [Daily]", "file_url": "ftp/25_Portfolios_5x5_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_100_size_and_bm_10_x_10_monthly", "dataset_name": "100 Portfolios Formed on Size and Book-to-Market (10 x 10)", "file_url": "ftp/100_Portfolios_10x10_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_100_size_and_bm_10_x_10_exdividends_monthly", "dataset_name": "100 Portfolios Formed on Size and Book-to-Market (10 x 10) [ex. Dividends]", "file_url": "ftp/100_Portfolios_10x10_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_100_size_and_bm_10_x_10_daily", "dataset_name": "100 Portfolios Formed on Size and Book-to-Market (10 x 10) [Daily]", "file_url": "ftp/100_Portfolios_10x10_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_op_2_x_3_monthly", "dataset_name": "6 Portfolios Formed on Size and Operating Profitability (2 x 3)", "file_url": "ftp/6_Portfolios_ME_OP_2x3_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_op_2_x_3_exdividends_monthly", "dataset_name": "6 Portfolios Formed on Size and Operating Profitability (2 x 3) [ex. Dividends]", "file_url": "ftp/6_Portfolios_ME_OP_2x3_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_op_2_x_3_daily", "dataset_name": "6 Portfolios Formed on Size and Operating Profitability (2 x 3) [Daily]", "file_url": "ftp/6_Portfolios_ME_OP_2x3_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_op_5_x_5_monthly", "dataset_name": "25 Portfolios Formed on Size and Operating Profitability (5 x 5)", "file_url": "ftp/25_Portfolios_ME_OP_5x5_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_op_5_x_5_exdividends_monthly", "dataset_name": "25 Portfolios Formed on Size and Operating Profitability (5 x 5) [ex. Dividends]", "file_url": "ftp/25_Portfolios_ME_OP_5x5_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_op_5_x_5_daily", "dataset_name": "25 Portfolios Formed on Size and Operating Profitability (5 x 5) [Daily]", "file_url": "ftp/25_Portfolios_ME_OP_5x5_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_100_size_and_op_10_x_10_monthly", "dataset_name": "100 Portfolios Formed on Size and Operating Profitability (10 x 10)", "file_url": "ftp/100_Portfolios_ME_OP_10x10_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_100_size_and_op_10_x_10_exdividends_monthly", "dataset_name": "100 Portfolios Formed on Size and Operating Profitability (10 x 10) [ex. Dividends]", "file_url": "ftp/100_Portfolios_10x10_ME_OP_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_100_size_and_op_10_x_10_daily", "dataset_name": "100 Portfolios Formed on Size and Operating Profitability (10 x 10) [Daily]", "file_url": "ftp/100_Portfolios_ME_OP_10x10_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_inv_2_x_3_monthly", "dataset_name": "6 Portfolios Formed on Size and Investment (2 x 3)", "file_url": "ftp/6_Portfolios_ME_INV_2x3_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_inv_2_x_3_exdividends_monthly", "dataset_name": "6 Portfolios Formed on Size and Investment (2 x 3) [ex. Dividends]", "file_url": "ftp/6_Portfolios_ME_INV_2x3_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_inv_2_x_3_daily", "dataset_name": "6 Portfolios Formed on Size and Investment (2 x 3) [Daily]", "file_url": "ftp/6_Portfolios_ME_INV_2x3_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_inv_5_x_5_monthly", "dataset_name": "25 Portfolios Formed on Size and Investment (5 x 5)", "file_url": "ftp/25_Portfolios_ME_INV_5x5_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_inv_5_x_5_exdividends_monthly", "dataset_name": "25 Portfolios Formed on Size and Investment (5 x 5) [ex. Dividends]", "file_url": "ftp/25_Portfolios_ME_INV_5x5_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_inv_5_x_5_daily", "dataset_name": "25 Portfolios Formed on Size and Investment (5 x 5) [Daily]", "file_url": "ftp/25_Portfolios_ME_INV_5x5_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_100_size_and_inv_10_x_10_monthly", "dataset_name": "100 Portfolios Formed on Size and Investment (10 x 10)", "file_url": "ftp/100_Portfolios_ME_INV_10x10_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_100_size_and_inv_10_x_10_exdividends_monthly", "dataset_name": "100 Portfolios Formed on Size and Investment (10 x 10) [ex. Dividends]", "file_url": "ftp/100_Portfolios_10x10_ME_INV_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_100_size_and_inv_10_x_10_daily", "dataset_name": "100 Portfolios Formed on Size and Investment (10 x 10) [Daily]", "file_url": "ftp/100_Portfolios_ME_INV_10x10_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_bm_and_op_5_x_5_monthly", "dataset_name": "25 Portfolios Formed on Book-to-Market and Operating Profitability (5 x 5)", "file_url": "ftp/25_Portfolios_BEME_OP_5x5_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_bm_and_op_5_x_5_exdividends_monthly", "dataset_name": "25 Portfolios Formed on Book-to-Market and Operating Profitability (5 x 5) [ex. Dividends]", "file_url": "ftp/25_Portfolios_BEME_OP_5x5_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_bm_and_op_5_x_5_daily", "dataset_name": "25 Portfolios Formed on Book-to-Market and Operating Profitability (5 x 5) [Daily]", "file_url": "ftp/25_Portfolios_BEME_OP_5x5_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_bm_and_inv_5_x_5_monthly", "dataset_name": "25 Portfolios Formed on Book-to-Market and Investment (5 x 5)", "file_url": "ftp/25_Portfolios_BEME_INV_5x5_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_bm_and_inv_5_x_5_exdividends_monthly", "dataset_name": "25 Portfolios Formed on Book-to-Market and Investment (5 x 5) [ex. Dividends]", "file_url": "ftp/25_Portfolios_BEME_INV_5x5_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_bm_and_inv_5_x_5_daily", "dataset_name": "25 Portfolios Formed on Book-to-Market and Investment (5 x 5) [Daily]", "file_url": "ftp/25_Portfolios_BEME_INV_5x5_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_op_and_inv_5_x_5_monthly", "dataset_name": "25 Portfolios Formed on Operating Profitability and Investment (5 x 5)", "file_url": "ftp/25_Portfolios_OP_INV_5x5_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_op_and_inv_5_x_5_exdividends_monthly", "dataset_name": "25 Portfolios Formed on Operating Profitability and Investment (5 x 5) [ex. Dividends]", "file_url": "ftp/25_Portfolios_OP_INV_5x5_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_op_and_inv_5_x_5_daily", "dataset_name": "25 Portfolios Formed on Operating Profitability and Investment (5 x 5) [Daily]", "file_url": "ftp/25_Portfolios_OP_INV_5x5_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_size_bm_and_op_2_x_4_x_4_monthly", "dataset_name": "32 Portfolios Formed on Size, Book-to-Market, and Operating Profitability (2 x 4 x 4)", "file_url": "ftp/32_Portfolios_ME_BEME_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_size_bm_and_op_2_x_4_x_4_exdividends_monthly", "dataset_name": "32 Portfolios Formed on Size, Book-to-Market, and Operating Profitability (2 x 4 x 4) [ex. Dividends]", "file_url": "ftp/32_Portfolios_ME_BEME_OP_2x4x4_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_size_bm_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 Portfolios Formed on Size, Book-to-Market, and Investment (2 x 4 x 4)", "file_url": "ftp/32_Portfolios_ME_BEME_INV_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_size_bm_and_inv_2_x_4_x_4_exdividends_monthly", "dataset_name": "32 Portfolios Formed on Size, Book-to-Market, and Investment (2 x 4 x 4) [ex. Dividends]", "file_url": "ftp/32_Portfolios_ME_BEME_INV_2x4x4_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_size_op_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 Portfolios Formed on Size, Operating Profitability, and Investment (2 x 4 x 4)", "file_url": "ftp/32_Portfolios_ME_OP_INV_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_size_op_and_inv_2_x_4_x_4_exdividends_monthly", "dataset_name": "32 Portfolios Formed on Size, Operating Profitability, and Investment (2 x 4 x 4) [ex. Dividends]", "file_url": "ftp/32_Portfolios_ME_OP_INV_2x4x4_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_earningsprice_monthly", "dataset_name": "Portfolios Formed on Earnings/Price", "file_url": "ftp/Portfolios_Formed_on_E-P_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_earningsprice_exdividends_monthly", "dataset_name": "Portfolios Formed on Earnings/Price [ex. Dividends]", "file_url": "ftp/Portfolios_Formed_on_E-P_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_cashflowprice_monthly", "dataset_name": "Portfolios Formed on Cashflow/Price", "file_url": "ftp/Portfolios_Formed_on_CF-P_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_cashflowprice_exdividends_monthly", "dataset_name": "Portfolios Formed on Cashflow/Price [ex. Dividends]", "file_url": "ftp/Portfolios_Formed_on_CF-P_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_dividend_yield_monthly", "dataset_name": "Portfolios Formed on Dividend Yield", "file_url": "ftp/Portfolios_Formed_on_D-P_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_dividend_yield_exdividends_monthly", "dataset_name": "Portfolios Formed on Dividend Yield [ex. Dividends]", "file_url": "ftp/Portfolios_Formed_on_D-P_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_earningsprice_monthly", "dataset_name": "6 Portfolios Formed on Size and Earnings/Price", "file_url": "ftp/6_Portfolios_ME_EP_2x3_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_earningsprice_exdividends_monthly", "dataset_name": "6 Portfolios Formed on Size and Earnings/Price [ex. Dividends]", "file_url": "ftp/6_Portfolios_ME_EP_2x3_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_cashflowprice_monthly", "dataset_name": "6 Portfolios Formed on Size and Cashflow/Price", "file_url": "ftp/6_Portfolios_ME_CFP_2x3_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_cashflowprice_exdividends_monthly", "dataset_name": "6 Portfolios Formed on Size and Cashflow/Price [ex. Dividends]", "file_url": "ftp/6_Portfolios_ME_CFP_2x3_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_dividend_yield_monthly", "dataset_name": "6 Portfolios Formed on Size and Dividend Yield", "file_url": "ftp/6_Portfolios_ME_DP_2x3_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_dividend_yield_exdividends_monthly", "dataset_name": "6 Portfolios Formed on Size and Dividend Yield [ex. Dividends]", "file_url": "ftp/6_Portfolios_ME_DP_2x3_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_momentum_factor_monthly", "dataset_name": "Momentum Factor (Mom)", "file_url": "ftp/F-F_Momentum_Factor_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_momentum_factor_daily", "dataset_name": "Momentum Factor (Mom) [Daily]", "file_url": "ftp/F-F_Momentum_Factor_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_momentum_2_x_3_monthly", "dataset_name": "6 Portfolios Formed on Size and Momentum (2 x 3)", "file_url": "ftp/6_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_momentum_2_x_3_daily", "dataset_name": "6 Portfolios Formed on Size and Momentum (2 x 3) [Daily]", "file_url": "ftp/6_Portfolios_ME_Prior_12_2_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_momentum_5_x_5_monthly", "dataset_name": "25 Portfolios Formed on Size and Momentum (5 x 5)", "file_url": "ftp/25_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_momentum_5_x_5_daily", "dataset_name": "25 Portfolios Formed on Size and Momentum (5 x 5) [Daily]", "file_url": "ftp/25_Portfolios_ME_Prior_12_2_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_10_momentum_monthly", "dataset_name": "10 Portfolios Formed on Momentum", "file_url": "ftp/10_Portfolios_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_10_momentum_daily", "dataset_name": "10 Portfolios Formed on Momentum [Daily]", "file_url": "ftp/10_Portfolios_Prior_12_2_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_shortterm_reversal_factor_st_rev_monthly", "dataset_name": "Short-Term Reversal Factor (ST Rev)", "file_url": "ftp/F-F_ST_Reversal_Factor_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_shortterm_reversal_factor_st_rev_daily", "dataset_name": "Short-Term Reversal Factor (ST Rev) [Daily]", "file_url": "ftp/F-F_ST_Reversal_Factor_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_shortterm_reversal_2_x_3_monthly", "dataset_name": "6 Portfolios Formed on Size and Short-Term Reversal (2 x 3)", "file_url": "ftp/6_Portfolios_ME_Prior_1_0_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_shortterm_reversal_2_x_3_daily", "dataset_name": "6 Portfolios Formed on Size and Short-Term Reversal (2 x 3) [Daily]", "file_url": "ftp/6_Portfolios_ME_Prior_1_0_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_shortterm_reversal_5_x_5_monthly", "dataset_name": "25 Portfolios Formed on Size and Short-Term Reversal (5 x 5)", "file_url": "ftp/25_Portfolios_ME_Prior_1_0_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_shortterm_reversal_5_x_5_daily", "dataset_name": "25 Portfolios Formed on Size and Short-Term Reversal (5 x 5) [Daily]", "file_url": "ftp/25_Portfolios_ME_Prior_1_0_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_10_shortterm_reversal_monthly", "dataset_name": "10 Portfolios Formed on Short-Term Reversal", "file_url": "ftp/10_Portfolios_Prior_1_0_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_10_shortterm_reversal_daily", "dataset_name": "10 Portfolios Formed on Short-Term Reversal [Daily]", "file_url": "ftp/10_Portfolios_Prior_1_0_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_longterm_reversal_factor_lt_rev_monthly", "dataset_name": "Long-Term Reversal Factor (LT Rev)", "file_url": "ftp/F-F_LT_Reversal_Factor_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_longterm_reversal_factor_lt_rev_daily", "dataset_name": "Long-Term Reversal Factor (LT Rev) [Daily]", "file_url": "ftp/F-F_LT_Reversal_Factor_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_longterm_reversal_2_x_3_monthly", "dataset_name": "6 Portfolios Formed on Size and Long-Term Reversal (2 x 3)", "file_url": "ftp/6_Portfolios_ME_Prior_60_13_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_size_and_longterm_reversal_2_x_3_daily", "dataset_name": "6 Portfolios Formed on Size and Long-Term Reversal (2 x 3) [Daily]", "file_url": "ftp/6_Portfolios_ME_Prior_60_13_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_longterm_reversal_5_x_5_monthly", "dataset_name": "25 Portfolios Formed on Size and Long-Term Reversal (5 x 5)", "file_url": "ftp/25_Portfolios_ME_Prior_60_13_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_longterm_reversal_5_x_5_daily", "dataset_name": "25 Portfolios Formed on Size and Long-Term Reversal (5 x 5) [Daily]", "file_url": "ftp/25_Portfolios_ME_Prior_60_13_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_10_longterm_reversal_monthly", "dataset_name": "10 Portfolios Formed on Long-Term Reversal", "file_url": "ftp/10_Portfolios_Prior_60_13_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_10_longterm_reversal_daily", "dataset_name": "10 Portfolios Formed on Long-Term Reversal [Daily]", "file_url": "ftp/10_Portfolios_Prior_60_13_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_accruals_monthly", "dataset_name": "Portfolios Formed on Accruals", "file_url": "ftp/Portfolios_Formed_on_AC_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_accruals_monthly", "dataset_name": "25 Portfolios Formed on Size and Accruals", "file_url": "ftp/25_Portfolios_ME_AC_5x5_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_market_beta_monthly", "dataset_name": "Portfolios Formed on Market Beta", "file_url": "ftp/Portfolios_Formed_on_BETA_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_market_beta_monthly", "dataset_name": "25 Portfolios Formed on Size and Market Beta", "file_url": "ftp/25_Portfolios_ME_BETA_5x5_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_net_share_issues_monthly", "dataset_name": "Portfolios Formed on Net Share Issues", "file_url": "ftp/Portfolios_Formed_on_NI_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_net_share_issues_monthly", "dataset_name": "25 Portfolios Formed on Size and Net Share Issues", "file_url": "ftp/25_Portfolios_ME_NI_5x5_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_variance_monthly", "dataset_name": "Portfolios Formed on Variance", "file_url": "ftp/Portfolios_Formed_on_VAR_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_variance_monthly", "dataset_name": "25 Portfolios Formed on Size and Variance", "file_url": "ftp/25_Portfolios_ME_VAR_5x5_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_residual_variance_monthly", "dataset_name": "Portfolios Formed on Residual Variance", "file_url": "ftp/Portfolios_Formed_on_RESVAR_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_size_and_residual_variance_monthly", "dataset_name": "25 Portfolios Formed on Size and Residual Variance", "file_url": "ftp/25_Portfolios_ME_RESVAR_5x5_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_5_industry_portfolios_monthly", "dataset_name": "5 Industry Portfolios", "file_url": "ftp/5_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_5_industry_portfolios_exdividends_monthly", "dataset_name": "5 Industry Portfolios [ex. Dividends]", "file_url": "ftp/5_Industry_Portfolios_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_5_industry_portfolios_daily", "dataset_name": "5 Industry Portfolios [Daily]", "file_url": "ftp/5_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_10_industry_portfolios_monthly", "dataset_name": "10 Industry Portfolios", "file_url": "ftp/10_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_10_industry_portfolios_exdividends_monthly", "dataset_name": "10 Industry Portfolios [ex. Dividends]", "file_url": "ftp/10_Industry_Portfolios_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_10_industry_portfolios_daily", "dataset_name": "10 Industry Portfolios [Daily]", "file_url": "ftp/10_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_12_industry_portfolios_monthly", "dataset_name": "12 Industry Portfolios", "file_url": "ftp/12_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_12_industry_portfolios_exdividends_monthly", "dataset_name": "12 Industry Portfolios [ex. Dividends]", "file_url": "ftp/12_Industry_Portfolios_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_12_industry_portfolios_daily", "dataset_name": "12 Industry Portfolios [Daily]", "file_url": "ftp/12_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_17_industry_portfolios_monthly", "dataset_name": "17 Industry Portfolios", "file_url": "ftp/17_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_17_industry_portfolios_exdividends_monthly", "dataset_name": "17 Industry Portfolios [ex. Dividends]", "file_url": "ftp/17_Industry_Portfolios_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_17_industry_portfolios_daily", "dataset_name": "17 Industry Portfolios [Daily]", "file_url": "ftp/17_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_30_industry_portfolios_monthly", "dataset_name": "30 Industry Portfolios", "file_url": "ftp/30_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_30_industry_portfolios_exdividends_monthly", "dataset_name": "30 Industry Portfolios [ex. Dividends]", "file_url": "ftp/30_Industry_Portfolios_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_30_industry_portfolios_daily", "dataset_name": "30 Industry Portfolios [Daily]", "file_url": "ftp/30_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_38_industry_portfolios_monthly", "dataset_name": "38 Industry Portfolios", "file_url": "ftp/38_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_38_industry_portfolios_exdividends_monthly", "dataset_name": "38 Industry Portfolios [ex. Dividends]", "file_url": "ftp/38_Industry_Portfolios_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_38_industry_portfolios_daily", "dataset_name": "38 Industry Portfolios [Daily]", "file_url": "ftp/38_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_48_industry_portfolios_monthly", "dataset_name": "48 Industry Portfolios", "file_url": "ftp/48_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_48_industry_portfolios_exdividends_monthly", "dataset_name": "48 Industry Portfolios [ex. Dividends]", "file_url": "ftp/48_Industry_Portfolios_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_48_industry_portfolios_daily", "dataset_name": "48 Industry Portfolios [Daily]", "file_url": "ftp/48_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_49_industry_portfolios_monthly", "dataset_name": "49 Industry Portfolios", "file_url": "ftp/49_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_49_industry_portfolios_exdividends_monthly", "dataset_name": "49 Industry Portfolios [ex. Dividends]", "file_url": "ftp/49_Industry_Portfolios_Wout_Div_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_49_industry_portfolios_daily", "dataset_name": "49 Industry Portfolios [Daily]", "file_url": "ftp/49_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_me_breakpoints_monthly", "dataset_name": "ME Breakpoints", "file_url": "ftp/ME_Breakpoints_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_beme_breakpoints_monthly", "dataset_name": "BE/ME Breakpoints", "file_url": "ftp/BE-ME_Breakpoints_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_op_breakpoints_monthly", "dataset_name": "Operating Profitability Breakpoints", "file_url": "ftp/OP_Breakpoints_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_inv_breakpoints_monthly", "dataset_name": "Investment Breakpoints", "file_url": "ftp/INV_Breakpoints_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_ep_breakpoints_monthly", "dataset_name": "E/P Breakpoints", "file_url": "ftp/E-P_Breakpoints_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_cfp_breakpoints_monthly", "dataset_name": "CF/P Breakpoints", "file_url": "ftp/CF-P_Breakpoints_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_dp_breakpoints_monthly", "dataset_name": "D/P Breakpoints", "file_url": "ftp/D-P_Breakpoints_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_prior_212_return_breakpoints_monthly", "dataset_name": "Prior (2-12) Return Breakpoints", "file_url": "ftp/Prior_2-12_Breakpoints_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_3_monthly", "dataset_name": "Fama/French Developed 3 Factors", "file_url": "ftp/Developed_3_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_3_daily", "dataset_name": "Fama/French Developed 3 Factors [Daily]", "file_url": "ftp/Developed_3_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_ex_us_3_monthly", "dataset_name": "Fama/French Developed ex US 3 Factors", "file_url": "ftp/Developed_ex_US_3_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_ex_us_3_daily", "dataset_name": "Fama/French Developed ex US 3 Factors [Daily]", "file_url": "ftp/Developed_ex_US_3_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_european_3_monthly", "dataset_name": "Fama/French European 3 Factors", "file_url": "ftp/Europe_3_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_european_3_daily", "dataset_name": "Fama/French European 3 Factors [Daily]", "file_url": "ftp/Europe_3_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_japanese_3_monthly", "dataset_name": "Fama/French Japanese 3 Factors", "file_url": "ftp/Japan_3_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_japanese_3_daily", "dataset_name": "Fama/French Japanese 3 Factors [Daily]", "file_url": "ftp/Japan_3_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_asia_pacific_ex_japan_3_monthly", "dataset_name": "Fama/French Asia Pacific ex Japan 3 Factors", "file_url": "ftp/Asia_Pacific_ex_Japan_3_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_asia_pacific_ex_japan_3_daily", "dataset_name": "Fama/French Asia Pacific ex Japan 3 Factors [Daily]", "file_url": "ftp/Asia_Pacific_ex_Japan_3_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_north_american_3_monthly", "dataset_name": "Fama/French North American 3 Factors", "file_url": "ftp/North_America_3_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_north_american_3_daily", "dataset_name": "Fama/French North American 3 Factors [Daily]", "file_url": "ftp/North_America_3_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_5_monthly", "dataset_name": "Fama/French Developed 5 Factors", "file_url": "ftp/Developed_5_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_5_daily", "dataset_name": "Fama/French Developed 5 Factors [Daily]", "file_url": "ftp/Developed_5_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_ex_us_5_monthly", "dataset_name": "Fama/French Developed ex US 5 Factors", "file_url": "ftp/Developed_ex_US_5_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_ex_us_5_daily", "dataset_name": "Fama/French Developed ex US 5 Factors [Daily]", "file_url": "ftp/Developed_ex_US_5_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_european_5_monthly", "dataset_name": "Fama/French European 5 Factors", "file_url": "ftp/Europe_5_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_european_5_daily", "dataset_name": "Fama/French European 5 Factors [Daily]", "file_url": "ftp/Europe_5_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_japanese_5_monthly", "dataset_name": "Fama/French Japanese 5 Factors", "file_url": "ftp/Japan_5_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_japanese_5_daily", "dataset_name": "Fama/French Japanese 5 Factors [Daily]", "file_url": "ftp/Japan_5_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_asia_pacific_ex_japan_5_monthly", "dataset_name": "Fama/French Asia Pacific ex Japan 5 Factors", "file_url": "ftp/Asia_Pacific_ex_Japan_5_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_asia_pacific_ex_japan_5_daily", "dataset_name": "Fama/French Asia Pacific ex Japan 5 Factors [Daily]", "file_url": "ftp/Asia_Pacific_ex_Japan_5_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_north_american_5_monthly", "dataset_name": "Fama/French North American 5 Factors", "file_url": "ftp/North_America_5_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_north_american_5_daily", "dataset_name": "Fama/French North American 5 Factors [Daily]", "file_url": "ftp/North_America_5_Factors_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_momentum_factor_monthly", "dataset_name": "Developed Momentum Factor (Mom)", "file_url": "ftp/Developed_Mom_Factor_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_momentum_factor_daily", "dataset_name": "Developed Momentum Factor (Mom) [Daily]", "file_url": "ftp/Developed_Mom_Factor_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_ex_us_momentum_factor_monthly", "dataset_name": "Developed ex US Momentum Factor (Mom)", "file_url": "ftp/Developed_ex_US_Mom_Factor_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_developed_ex_us_momentum_factor_daily", "dataset_name": "Developed ex US Momentum Factor (Mom) [Daily]", "file_url": "ftp/Developed_ex_US_Mom_Factor_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_european_momentum_factor_monthly", "dataset_name": "European Momentum Factor (Mom)", "file_url": "ftp/Europe_Mom_Factor_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_european_momentum_factor_daily", "dataset_name": "European Momentum Factor (Mom) [Daily]", "file_url": "ftp/Europe_Mom_Factor_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_japanese_momentum_factor_monthly", "dataset_name": "Japanese Momentum Factor (Mom)", "file_url": "ftp/Japan_Mom_Factor_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_japanese_momentum_factor_daily", "dataset_name": "Japanese Momentum Factor (Mom) [Daily]", "file_url": "ftp/Japan_Mom_Factor_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_asia_pacific_ex_japan_momentum_factor_monthly", "dataset_name": "Asia Pacific ex Japan Momentum Factor (Mom)", "file_url": "ftp/Asia_Pacific_ex_Japan_MOM_Factor_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_asia_pacific_ex_japan_momentum_factor_daily", "dataset_name": "Asia Pacific ex Japan Momentum Factor (Mom) [Daily]", "file_url": "ftp/Asia_Pacific_ex_Japan_MOM_Factor_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_north_american_momentum_factor_monthly", "dataset_name": "North American Momentum Factor (Mom)", "file_url": "ftp/North_America_Mom_Factor_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_north_american_momentum_factor_daily", "dataset_name": "North American Momentum Factor (Mom) [Daily]", "file_url": "ftp/North_America_Mom_Factor_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_size_and_bm_2_x_3_monthly", "dataset_name": "6 Developed Portfolios Formed on Size and Book-to-Market (2 x 3)", "file_url": "ftp/Developed_6_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_size_and_bm_2_x_3_daily", "dataset_name": "6 Developed Portfolios Formed on Size and Book-to-Market (2 x 3) [Daily]", "file_url": "ftp/Developed_6_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_ex_us_size_and_bm_2_x_3_monthly", "dataset_name": "6 Developed ex US Portfolios Formed on Size and Book-to-Market (2 x 3)", "file_url": "ftp/Developed_ex_US_6_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_ex_us_size_and_bm_2_x_3_daily", "dataset_name": "6 Developed ex US Portfolios Formed on Size and Book-to-Market (2 x 3) [Daily]", "file_url": "ftp/Developed_ex_US_6_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_european_size_and_bm_2_x_3_monthly", "dataset_name": "6 European Portfolios Formed on Size and Book-to-Market (2 x 3)", "file_url": "ftp/Europe_6_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_european_size_and_bm_2_x_3_daily", "dataset_name": "6 European Portfolios Formed on Size and Book-to-Market (2 x 3) [Daily]", "file_url": "ftp/Europe_6_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_japanese_size_and_bm_2_x_3_monthly", "dataset_name": "6 Japanese Portfolios Formed on Size and Book-to-Market (2 x 3)", "file_url": "ftp/Japan_6_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_japanese_size_and_bm_2_x_3_daily", "dataset_name": "6 Japanese Portfolios Formed on Size and Book-to-Market (2 x 3) [Daily]", "file_url": "ftp/Japan_6_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_asia_pacific_ex_japan_size_and_bm_2_x_3_monthly", "dataset_name": "6 Asia Pacific ex Japan Portfolios Formed on Size and Book-to-Market (2 x 3)", "file_url": "ftp/Asia_Pacific_ex_Japan_6_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_asia_pacific_ex_japan_size_and_bm_2_x_3_daily", "dataset_name": "6 Asia Pacific ex Japan Portfolios Formed on Size and Book-to-Market (2 x 3) [Daily]", "file_url": "ftp/Asia_Pacific_ex_Japan_6_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_north_american_size_and_bm_2_x_3_monthly", "dataset_name": "6 North American Portfolios Formed on Size and Book-to-Market (2 x 3)", "file_url": "ftp/North_America_6_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_north_american_size_and_bm_2_x_3_daily", "dataset_name": "6 North American Portfolios Formed on Size and Book-to-Market (2 x 3) [Daily]", "file_url": "ftp/North_America_6_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_size_and_bm_5_x_5_monthly", "dataset_name": "25 Developed Portfolios Formed on Size and Book-to-Market (5 x 5)", "file_url": "ftp/Developed_25_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_size_and_bm_5_x_5_daily", "dataset_name": "25 Developed Portfolios Formed on Size and Book-to-Market (5 x 5) [Daily]", "file_url": "ftp/Developed_25_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_ex_us_size_and_bm_5_x_5_monthly", "dataset_name": "25 Developed ex US Portfolios Formed on Size and Book-to-Market (5 x 5)", "file_url": "ftp/Developed_ex_US_25_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_ex_us_size_and_bm_5_x_5_daily", "dataset_name": "25 Developed ex US Portfolios Formed on Size and Book-to-Market (5 x 5) [Daily]", "file_url": "ftp/Developed_ex_US_25_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_european_size_and_bm_5_x_5_monthly", "dataset_name": "25 European Portfolios Formed on Size and Book-to-Market (5 x 5)", "file_url": "ftp/Europe_25_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_european_size_and_bm_5_x_5_daily", "dataset_name": "25 European Portfolios Formed on Size and Book-to-Market (5 x 5) [Daily]", "file_url": "ftp/Europe_25_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_japanese_size_and_bm_5_x_5_monthly", "dataset_name": "25 Japanese Portfolios Formed on Size and Book-to-Market (5 x 5)", "file_url": "ftp/Japan_25_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_japanese_size_and_bm_5_x_5_daily", "dataset_name": "25 Japanese Portfolios Formed on Size and Book-to-Market (5 x 5) [Daily]", "file_url": "ftp/Japan_25_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_asia_pacific_ex_japan_size_and_bm_5_x_5_monthly", "dataset_name": "25 Asia Pacific ex Japan Portfolios Formed on Size and Book-to-Market (5 x 5)", "file_url": "ftp/Asia_Pacific_ex_Japan_25_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_asia_pacific_ex_japan_size_and_bm_5_x_5_daily", "dataset_name": "25 Asia Pacific ex Japan Portfolios Formed on Size and Book-to-Market (5 x 5) [Daily]", "file_url": "ftp/Asia_Pacific_ex_Japan_25_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_north_american_size_and_bm_5_x_5_monthly", "dataset_name": "25 North American Portfolios Formed on Size and Book-to-Market (5 x 5)", "file_url": "ftp/North_America_25_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_north_american_size_and_bm_5_x_5_daily", "dataset_name": "25 North American Portfolios Formed on Size and Book-to-Market (5 x 5) [Daily]", "file_url": "ftp/North_America_25_Portfolios_ME_BE-ME_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_size_and_op_2_x_3_monthly", "dataset_name": "6 Developed Portfolios Formed on Size and Operating Profitability (2 x 3)", "file_url": "ftp/Developed_6_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_size_and_op_2_x_3_daily", "dataset_name": "6 Developed Portfolios Formed on Size and Operating Profitability (2 x 3) [Daily]", "file_url": "ftp/Developed_6_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_ex_us_size_and_op_2_x_3_monthly", "dataset_name": "6 Developed ex US Portfolios Formed on Size and Operating Profitability (2 x 3)", "file_url": "ftp/Developed_ex_US_6_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_ex_us_size_and_op_2_x_3_daily", "dataset_name": "6 Developed ex US Portfolios Formed on Size and Operating Profitability (2 x 3) [Daily]", "file_url": "ftp/Developed_ex_US_6_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_european_size_and_op_2_x_3_monthly", "dataset_name": "6 European Portfolios Formed on Size and Operating Profitability (2 x 3)", "file_url": "ftp/Europe_6_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_european_size_and_op_2_x_3_daily", "dataset_name": "6 European Portfolios Formed on Size and Operating Profitability (2 x 3) [Daily]", "file_url": "ftp/Europe_6_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_japanese_size_and_op_2_x_3_monthly", "dataset_name": "6 Japanese Portfolios Formed on Size and Operating Profitability (2 x 3)", "file_url": "ftp/Japan_6_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_japanese_size_and_op_2_x_3_daily", "dataset_name": "6 Japanese Portfolios Formed on Size and Operating Profitability (2 x 3) [Daily]", "file_url": "ftp/Japan_6_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_asia_pacific_ex_japan_size_and_op_2_x_3_monthly", "dataset_name": "6 Asia Pacific ex Japan Portfolios Formed on Size and Operating Profitability (2 x 3)", "file_url": "ftp/Asia_Pacific_ex_Japan_6_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_asia_pacific_ex_japan_size_and_op_2_x_3_daily", "dataset_name": "6 Asia Pacific ex Japan Portfolios Formed on Size and Operating Profitability (2 x 3) [Daily]", "file_url": "ftp/Asia_Pacific_ex_Japan_6_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_north_american_size_and_op_2_x_3_monthly", "dataset_name": "6 North American Portfolios Formed on Size and Operating Profitability (2 x 3)", "file_url": "ftp/North_America_6_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_north_american_size_and_op_2_x_3_daily", "dataset_name": "6 North American Portfolios Formed on Size and Operating Profitability (2 x 3) [Daily]", "file_url": "ftp/North_America_6_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_size_and_op_5_x_5_monthly", "dataset_name": "25 Developed Portfolios Formed on Size and Operating Profitability (5 x 5)", "file_url": "ftp/Developed_25_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_size_and_op_5_x_5_daily", "dataset_name": "25 Developed Portfolios Formed on Size and Operating Profitability (5 x 5) [Daily]", "file_url": "ftp/Developed_25_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_ex_us_size_and_op_5_x_5_monthly", "dataset_name": "25 Developed ex US Portfolios Formed on Size and Operating Profitability (5 x 5)", "file_url": "ftp/Developed_ex_US_25_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_ex_us_size_and_op_5_x_5_daily", "dataset_name": "25 Developed ex US Portfolios Formed on Size and Operating Profitability (5 x 5) [Daily]", "file_url": "ftp/Developed_ex_US_25_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_european_size_and_op_5_x_5_monthly", "dataset_name": "25 European Portfolios Formed on Size and Operating Profitability (5 x 5)", "file_url": "ftp/Europe_25_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_european_size_and_op_5_x_5_daily", "dataset_name": "25 European Portfolios Formed on Size and Operating Profitability (5 x 5) [Daily]", "file_url": "ftp/Europe_25_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_japanese_size_and_op_5_x_5_monthly", "dataset_name": "25 Japanese Portfolios Formed on Size and Operating Profitability (5 x 5)", "file_url": "ftp/Japan_25_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_japanese_size_and_op_5_x_5_daily", "dataset_name": "25 Japanese Portfolios Formed on Size and Operating Profitability (5 x 5) [Daily]", "file_url": "ftp/Japan_25_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_asia_pacific_ex_japan_size_and_op_5_x_5_monthly", "dataset_name": "25 Asia Pacific ex Japan Portfolios Formed on Size and Operating Profitability (5 x 5)", "file_url": "ftp/Asia_Pacific_ex_Japan_25_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_asia_pacific_ex_japan_size_and_op_5_x_5_daily", "dataset_name": "25 Asia Pacific ex Japan Portfolios Formed on Size and Operating Profitability (5 x 5) [Daily]", "file_url": "ftp/Asia_Pacific_ex_Japan_25_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_north_american_size_and_op_5_x_5_monthly", "dataset_name": "25 North American Portfolios Formed on Size and Operating Profitability (5 x 5)", "file_url": "ftp/North_America_25_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_north_american_size_and_op_5_x_5_daily", "dataset_name": "25 North American Portfolios Formed on Size and Operating Profitability (5 x 5) [Daily]", "file_url": "ftp/North_America_25_Portfolios_ME_OP_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_size_and_inv_2_x_3_monthly", "dataset_name": "6 Developed Portfolios Formed on Size and Investment (2 x 3)", "file_url": "ftp/Developed_6_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_size_and_inv_2_x_3_daily", "dataset_name": "6 Developed Portfolios Formed on Size and Investment (2 x 3) [Daily]", "file_url": "ftp/Developed_6_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_ex_us_size_and_inv_2_x_3_monthly", "dataset_name": "6 Developed ex US Portfolios Formed on Size and Investment (2 x 3)", "file_url": "ftp/Developed_ex_US_6_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_ex_us_size_and_inv_2_x_3_daily", "dataset_name": "6 Developed ex US Portfolios Formed on Size and Investment (2 x 3) [Daily]", "file_url": "ftp/Developed_ex_US_6_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_european_size_and_inv_2_x_3_monthly", "dataset_name": "6 European Portfolios Formed on Size and Investment (2 x 3)", "file_url": "ftp/Europe_6_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_european_size_and_inv_2_x_3_daily", "dataset_name": "6 European Portfolios Formed on Size and Investment (2 x 3) [Daily]", "file_url": "ftp/Europe_6_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_japanese_size_and_inv_2_x_3_monthly", "dataset_name": "6 Japanese Portfolios Formed on Size and Investment (2 x 3)", "file_url": "ftp/Japan_6_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_japanese_size_and_inv_2_x_3_daily", "dataset_name": "6 Japanese Portfolios Formed on Size and Investment (2 x 3) [Daily]", "file_url": "ftp/Japan_6_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_asia_pacific_ex_japan_size_and_inv_2_x_3_monthly", "dataset_name": "6 Asia Pacific ex Japan Portfolios Formed on Size and Investment (2 x 3)", "file_url": "ftp/Asia_Pacific_ex_Japan_6_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_asia_pacific_ex_japan_size_and_inv_2_x_3_daily", "dataset_name": "6 Asia Pacific ex Japan Portfolios Formed on Size and Investment (2 x 3) [Daily]", "file_url": "ftp/Asia_Pacific_ex_Japan_6_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_north_american_size_and_inv_2_x_3_monthly", "dataset_name": "6 North American Portfolios Formed on Size and Investment (2 x 3)", "file_url": "ftp/North_America_6_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_north_american_size_and_inv_2_x_3_daily", "dataset_name": "6 North American Portfolios Formed on Size and Investment (2 x 3) [Daily]", "file_url": "ftp/North_America_6_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_size_and_inv_5_x_5_monthly", "dataset_name": "25 Developed Portfolios Formed on Size and Investment (5 x 5)", "file_url": "ftp/Developed_25_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_size_and_inv_5_x_5_daily", "dataset_name": "25 Developed Portfolios Formed on Size and Investment (5 x 5) [Daily]", "file_url": "ftp/Developed_25_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_ex_us_size_and_inv_5_x_5_monthly", "dataset_name": "25 Developed ex US Portfolios Formed on Size and Investment (5 x 5)", "file_url": "ftp/Developed_ex_US_25_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_ex_us_size_and_inv_5_x_5_daily", "dataset_name": "25 Developed ex US Portfolios Formed on Size and Investment (5 x 5) [Daily]", "file_url": "ftp/Developed_ex_US_25_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_european_size_and_inv_5_x_5_monthly", "dataset_name": "25 European Portfolios Formed on Size and Investment (5 x 5)", "file_url": "ftp/Europe_25_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_european_size_and_inv_5_x_5_daily", "dataset_name": "25 European Portfolios Formed on Size and Investment (5 x 5) [Daily]", "file_url": "ftp/Europe_25_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_japanese_size_and_inv_5_x_5_monthly", "dataset_name": "25 Japanese Portfolios Formed on Size and Investment (5 x 5)", "file_url": "ftp/Japan_25_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_japanese_size_and_inv_5_x_5_daily", "dataset_name": "25 Japanese Portfolios Formed on Size and Investment (5 x 5) [Daily]", "file_url": "ftp/Japan_25_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_asia_pacific_ex_japan_size_and_inv_5_x_5_monthly", "dataset_name": "25 Asia Pacific ex Japan Portfolios Formed on Size and Investment (5 x 5)", "file_url": "ftp/Asia_Pacific_ex_Japan_25_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_asia_pacific_ex_japan_size_and_inv_5_x_5_daily", "dataset_name": "25 Asia Pacific ex Japan Portfolios Formed on Size and Investment (5 x 5) [Daily]", "file_url": "ftp/Asia_Pacific_ex_Japan_25_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_north_american_size_and_inv_5_x_5_monthly", "dataset_name": "25 North American Portfolios Formed on Size and Investment (5 x 5)", "file_url": "ftp/North_America_25_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_north_american_size_and_inv_5_x_5_daily", "dataset_name": "25 North American Portfolios Formed on Size and Investment (5 x 5) [Daily]", "file_url": "ftp/North_America_25_Portfolios_ME_INV_Daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_size_and_momentum_2_x_3_monthly", "dataset_name": "6 Developed Portfolios Formed on Size and Momentum (2 x 3)", "file_url": "ftp/Developed_6_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_size_and_momentum_2_x_3_daily", "dataset_name": "6 Developed Portfolios Formed on Size and Momentum (2 x 3) [Daily]", "file_url": "ftp/Developed_6_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_ex_us_size_and_momentum_2_x_3_monthly", "dataset_name": "6 Developed ex US Portfolios Formed on Size and Momentum (2 x 3)", "file_url": "ftp/Developed_ex_US_6_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_developed_ex_us_size_and_momentum_2_x_3_daily", "dataset_name": "6 Developed ex US Portfolios Formed on Size and Momentum (2 x 3) [Daily]", "file_url": "ftp/Developed_ex_US_6_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_european_size_and_momentum_2_x_3_monthly", "dataset_name": "6 European Portfolios Formed on Size and Momentum (2 x 3)", "file_url": "ftp/Europe_6_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_european_size_and_momentum_2_x_3_daily", "dataset_name": "6 European Portfolios Formed on Size and Momentum (2 x 3) [Daily]", "file_url": "ftp/Europe_6_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_japanese_size_and_momentum_2_x_3_monthly", "dataset_name": "6 Japanese Portfolios Formed on Size and Momentum (2 x 3)", "file_url": "ftp/Japan_6_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_japanese_size_and_momentum_2_x_3_daily", "dataset_name": "6 Japanese Portfolios Formed on Size and Momentum (2 x 3) [Daily]", "file_url": "ftp/Japan_6_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_asia_pacific_ex_japan_size_and_momentum_2_x_3_monthly", "dataset_name": "6 Asia Pacific ex Japan Portfolios Formed on Size and Momentum (2 x 3)", "file_url": "ftp/Asia_Pacific_ex_Japan_6_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_asia_pacific_ex_japan_size_and_momentum_2_x_3_daily", "dataset_name": "6 Asia Pacific ex Japan Portfolios Formed on Size and Momentum (2 x 3) [Daily]", "file_url": "ftp/Asia_Pacific_ex_Japan_6_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_north_american_size_and_momentum_2_x_3_monthly", "dataset_name": "6 North American Portfolios Formed on Size and Momentum (2 x 3)", "file_url": "ftp/North_America_6_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_north_american_size_and_momentum_2_x_3_daily", "dataset_name": "6 North American Portfolios Formed on Size and Momentum (2 x 3) [Daily]", "file_url": "ftp/North_America_6_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_size_and_momentum_5_x_5_monthly", "dataset_name": "25 Developed Portfolios Formed on Size and Momentum (5 x 5)", "file_url": "ftp/Developed_25_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_size_and_momentum_5_x_5_daily", "dataset_name": "25 Developed Portfolios Formed on Size and Momentum (5 x 5) [Daily]", "file_url": "ftp/Developed_25_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_ex_us_size_and_momentum_5_x_5_monthly", "dataset_name": "25 Developed ex US Portfolios Formed on Size and Momentum (5 x 5)", "file_url": "ftp/Developed_ex_US_25_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_developed_ex_us_size_and_momentum_5_x_5_daily", "dataset_name": "25 Developed ex US Portfolios Formed on Size and Momentum (5 x 5) [Daily]", "file_url": "ftp/Developed_ex_US_25_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_european_size_and_momentum_5_x_5_monthly", "dataset_name": "25 European Portfolios Formed on Size and Momentum (5 x 5)", "file_url": "ftp/Europe_25_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_european_size_and_momentum_5_x_5_daily", "dataset_name": "25 European Portfolios Formed on Size and Momentum (5 x 5) [Daily]", "file_url": "ftp/Europe_25_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_japanese_size_and_momentum_5_x_5_monthly", "dataset_name": "25 Japanese Portfolios Formed on Size and Momentum (5 x 5)", "file_url": "ftp/Japan_25_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_japanese_size_and_momentum_5_x_5_daily", "dataset_name": "25 Japanese Portfolios Formed on Size and Momentum (5 x 5) [Daily]", "file_url": "ftp/Japan_25_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_asia_pacific_ex_japan_size_and_momentum_5_x_5_monthly", "dataset_name": "25 Asia Pacific ex Japan Portfolios Formed on Size and Momentum (5 x 5)", "file_url": "ftp/Asia_Pacific_ex_Japan_25_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_asia_pacific_ex_japan_size_and_momentum_5_x_5_daily", "dataset_name": "25 Asia Pacific ex Japan Portfolios Formed on Size and Momentum (5 x 5) [Daily]", "file_url": "ftp/Asia_Pacific_ex_Japan_25_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_north_american_size_and_momentum_5_x_5_monthly", "dataset_name": "25 North American Portfolios Formed on Size and Momentum (5 x 5)", "file_url": "ftp/North_America_25_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_25_north_american_size_and_momentum_5_x_5_daily", "dataset_name": "25 North American Portfolios Formed on Size and Momentum (5 x 5) [Daily]", "file_url": "ftp/North_America_25_Portfolios_ME_Prior_250_20_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_developed_size_bm_and_op_2_x_4_x_4_monthly", "dataset_name": "32 Developed Portfolios Formed on Size, Book-to-Market, and Operating Profitability (2 x 4 x 4)", "file_url": "ftp/Developed_32_Portfolios_ME_BE-ME_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_developed_ex_us_size_bm_and_op_2_x_4_x_4_monthly", "dataset_name": "32 Developed ex US Portfolios Formed on Size, Book-to-Market, and Operating Profitability (2 x 4 x 4)", "file_url": "ftp/Developed_ex_US_32_Portfolios_ME_BE-ME_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_european_size_bm_and_op_2_x_4_x_4_monthly", "dataset_name": "32 European Portfolios Formed on Size, Book-to-Market, and Operating Profitability (2 x 4 x 4)", "file_url": "ftp/Europe_32_Portfolios_ME_BE-ME_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_japanese_size_bm_and_op_2_x_4_x_4_monthly", "dataset_name": "32 Japanese Portfolios Formed on Size, Book-to-Market, and Operating Profitability (2 x 4 x 4)", "file_url": "ftp/Japan_32_Portfolios_ME_BE-ME_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_asia_pacific_ex_japan_size_bm_and_op_2_x_4_x_4_monthly", "dataset_name": "32 Asia Pacific ex Japan Portfolios Formed on Size, Book-to-Market, and Operating Profitability (2 x 4 x 4)", "file_url": "ftp/Asia_Pacific_ex_Japan_32_Portfolios_ME_BE-ME_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_north_american_size_bm_and_op_2_x_4_x_4_monthly", "dataset_name": "32 North American Portfolios Formed on Size, Book-to-Market, and Operating Profitability (2 x 4 x 4)", "file_url": "ftp/North_America_32_Portfolios_ME_BE-ME_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_developed_size_bm_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 Developed Portfolios Formed on Size, Book-to-Market, and Investment (2 x 4 x 4)", "file_url": "ftp/Developed_32_Portfolios_ME_BE-ME_INV(TA)_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_developed_ex_us_size_bm_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 Developed ex US Portfolios Formed on Size, Book-to-Market, and Investment (2 x 4 x 4)", "file_url": "ftp/Developed_ex_US_32_Portfolios_ME_BE-ME_INV(TA)_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_european_size_bm_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 European Portfolios Formed on Size, Book-to-Market, and Investment (2 x 4 x 4)", "file_url": "ftp/Europe_32_Portfolios_ME_BE-ME_INV(TA)_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_japanese_size_bm_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 Japanese Portfolios Formed on Size, Book-to-Market, and Investment (2 x 4 x 4)", "file_url": "ftp/Japan_32_Portfolios_ME_BE-ME_INV(TA)_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_asia_pacific_ex_japan_size_bm_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 Asia Pacific ex Japan Portfolios Formed on Size, Book-to-Market, and Investment (2 x 4 x 4)", "file_url": "ftp/Asia_Pacific_ex_Japan_32_Portfolios_ME_BE-ME_INV(TA)_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_north_american_size_bm_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 North American Portfolios Formed on Size, Book-to-Market, and Investment (2 x 4 x 4)", "file_url": "ftp/North_America_32_Portfolios_ME_BE-ME_INV(TA)_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_developed_size_op_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 Developed Portfolios Formed on Size, Operating Profitability, and Investment (2 x 4 x 4)", "file_url": "ftp/Developed_32_Portfolios_ME_INV(TA)_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_developed_ex_us_size_op_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 Developed ex US Portfolios Formed on Size, Operating Profitability, and Investment (2 x 4 x 4)", "file_url": "ftp/Developed_ex_US_32_Portfolios_ME_INV(TA)_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_european_size_op_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 European Portfolios Formed on Size, Operating Profitability, and Investment (2 x 4 x 4)", "file_url": "ftp/Europe_32_Portfolios_ME_INV(TA)_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_japanese_size_op_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 Japanese Portfolios Formed on Size, Operating Profitability, and Investment (2 x 4 x 4)", "file_url": "ftp/Japan_32_Portfolios_ME_INV(TA)_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_asia_pacific_ex_japan_size_op_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 Asia Pacific ex Japan Portfolios Formed on Size, Operating Profitability, and Investment (2 x 4 x 4)", "file_url": "ftp/Asia_Pacific_ex_Japan_32_Portfolios_ME_INV(TA)_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_32_north_american_size_op_and_inv_2_x_4_x_4_monthly", "dataset_name": "32 North American Portfolios Formed on Size, Operating Profitability, and Investment (2 x 4 x 4)", "file_url": "ftp/North_America_32_Portfolios_ME_INV(TA)_OP_2x4x4_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_emerging_5_monthly", "dataset_name": "Fama/French Emerging 5 Factors", "file_url": "ftp/Emerging_5_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_emerging_momentum_factor_monthly", "dataset_name": "Emerging Momentum Factor (Mom)", "file_url": "ftp/Emerging_MOM_Factor_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_emerging_market_size_and_bm_2_x_3_monthly", "dataset_name": "6 Emerging Market Portfolios Formed on Size and Book-to-Market (2 x 3)", "file_url": "ftp/Emerging_Markets_6_Portfolios_ME_BE-ME_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_emerging_market_size_and_op_2_x_3_monthly", "dataset_name": "6 Emerging Market Portfolios Formed on Size and Operating Profitability (2 x 3)", "file_url": "ftp/Emerging_Markets_6_Portfolios_ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_emerging_market_size_and_inv_2_x_3_monthly", "dataset_name": "6 Emerging Market Portfolios Formed on Size and Investment (2 x 3)", "file_url": "ftp/Emerging_Markets_6_Portfolios_ME_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_6_emerging_market_size_and_momentum_2_x_3_monthly", "dataset_name": "6 Emerging Market Portfolios Formed on Size and Momentum (2 x 3)", "file_url": "ftp/Emerging_Markets_6_Portfolios_ME_Prior_12_2_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_4_emerging_market_bm__and_op_2_x_2___monthly", "dataset_name": "4 Emerging Market Portfolios Formed on Book-to-Market  and Operating Profitability (2 x 2)  ", "file_url": "ftp/Emerging_Markets_4_Portfolios_BE-ME_OP_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_4_emerging_market_op_and_inv_2_x_2_monthly", "dataset_name": "4 Emerging Market Portfolios Formed on Operating Profitability and Investment (2 x 2)", "file_url": "ftp/Emerging_Markets_4_Portfolios_OP_INV_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_4_emerging_market_bm_and_inv_2_x_2_monthly", "dataset_name": "4 Emerging Market Portfolios Formed on Book-to-Market and Investment (2 x 2)", "file_url": "ftp/Emerging_Markets_4_Portfolios_BE-ME_INV_CSV.zip", "domain": "Fama-French"},
]

# %% Supported legacy Fama-French datasets
# List of dicts of the legacy names of initially supported Fama-French
# datasets. Not included in the exported list_supported_datasets()
# function.
_FF_LEGACY_DATASETS = [
    {"type": "factors_ff3_daily", "dataset_name": "Fama/French 3 Factors [Daily]", "file_url": "ftp/F-F_Research_Data_Factors_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff3_weekly", "dataset_name": "Fama/French 3 Factors [Weekly]", "file_url": "ftp/F-F_Research_Data_Factors_weekly_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff3_monthly", "dataset_name": "Fama/French 3 Factors", "file_url": "ftp/F-F_Research_Data_Factors_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff5_daily", "dataset_name": "Fama/French 5 Factors (2x3) [Daily]", "file_url": "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff5_monthly", "dataset_name": "Fama/French 5 Factors (2x3)", "file_url": "ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_5_monthly", "dataset_name": "5 Industry Portfolios", "file_url": "ftp/5_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_5_daily", "dataset_name": "5 Industry Portfolios [Daily]", "file_url": "ftp/5_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_10_monthly", "dataset_name": "10 Industry Portfolios", "file_url": "ftp/10_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_10_daily", "dataset_name": "10 Industry Portfolios [Daily]", "file_url": "ftp/10_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_30_monthly", "dataset_name": "30 Industry Portfolios", "file_url": "ftp/30_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_30_daily", "dataset_name": "30 Industry Portfolios [Daily]", "file_url": "ftp/30_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_38_monthly", "dataset_name": "38 Industry Portfolios", "file_url": "ftp/38_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_38_daily", "dataset_name": "38 Industry Portfolios [Daily]", "file_url": "ftp/38_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_48_monthly", "dataset_name": "48 Industry Portfolios", "file_url": "ftp/48_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_48_daily", "dataset_name": "48 Industry Portfolios [Daily]", "file_url": "ftp/48_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_49_monthly", "dataset_name": "49 Industry Portfolios", "file_url": "ftp/49_Industry_Portfolios_CSV.zip", "domain": "Fama-French"},
    {"type": "factors_ff_industry_49_daily", "dataset_name": "49 Industry Portfolios [Daily]", "file_url": "ftp/49_Industry_Portfolios_daily_CSV.zip", "domain": "Fama-French"},
]

# %% Supported Global Q datasets
# List of dicts of the supported Global Q datasets, including their
# names and frequencies (daily, weekly, weekly week-to-week, monthly,
# quarterly, annual). Each dataset type is associated with the Global Q
# model, specifically the q5 factors model for the year 2023. Annotated
# with the domain "Global Q".
_Q_DATASETS = [
    {"type": "factors_q5_daily", "dataset_name": "q5_factors_daily_2024", "domain": "Global Q"},
    {"type": "factors_q5_weekly", "dataset_name": "q5_factors_weekly_2024", "domain": "Global Q"},
    {"type": "factors_q5_weekly_w2w", "dataset_name": "q5_factors_weekly_w2w_2024", "domain": "Global Q"},
    {"type": "factors_q5_monthly", "dataset_name": "q5_factors_monthly_2024", "domain": "Global Q"},
    {"type": "factors_q5_quarterly", "dataset_name": "q5_factors_quarterly_2024", "domain": "Global Q"},
    {"type": "factors_q5_annual", "dataset_name": "q5_factors_annual_2024", "domain": "Global Q"},
]

# %% Supported macro predictor datasets
# List of dicts of the supported macro predictor datasets provided by
# Goyal-Welch, including their frequencies (monthly, quarterly,
# annual). All datasets reference the same source file,
# "PredictorData2022.xlsx" for the year 2022. Annotated with the
# domain "Goyal-Welch".
_MACRO_DATASETS = [
    {"type": "macro_predictors_monthly", "dataset_name": "PredictorData2022.xlsx", "domain": "Goyal-Welch"},
    {"type": "macro_predictors_quarterly", "dataset_name": "PredictorData2022.xlsx", "domain": "Goyal-Welch"},
    {"type": "macro_predictors_annual", "dataset_name": "PredictorData2022.xlsx", "domain": "Goyal-Welch"},
]

# %% Supported WRDS datasets
# List of dicts of the supported datasets provided via WRDS. Annotated
# with the domain "WRDS".
_WRDS_DATASETS = [
    {"type": "wrds_crsp_monthly", "dataset_name": "crsp.msf, crsp.msenames, crsp.msedelist", "domain": "WRDS"},
    {"type": "wrds_crsp_daily", "dataset_name": "crsp.dsf, crsp.msenames, crsp.msedelist", "domain": "WRDS"},
    {"type": "wrds_compustat_annual", "dataset_name": "comp.funda", "domain": "WRDS"},
    {"type": "wrds_compustat_quarterly", "dataset_name": "comp.fundq", "domain": "WRDS"},
    {"type": "wrds_ccm_links", "dataset_name": "crsp.ccmxpf_linktable", "domain": "WRDS"},
    {"type": "wrds_fisd", "dataset_name": "fisd.fisd_mergedissue, fisd.fisd_mergedissuer", "domain": "WRDS"},
    {"type": "wrds_trace_enhanced", "dataset_name": "trace.trace_enhanced", "domain": "WRDS"},
]

# %% Supported pseudo WRDS datasets
# List of dicts of the supported pseudo WRDS datasets generated by
# download_data(domain="Pseudo Data", ...). Annotated with the domain
# "Pseudo Data".
_PSEUDO_DATASETS = [
    {"type": "crsp_monthly", "dataset_name": "pseudo crsp_monthly", "domain": "Pseudo Data"},
    {"type": "crsp_daily", "dataset_name": "pseudo crsp_daily", "domain": "Pseudo Data"},
    {"type": "compustat_annual", "dataset_name": "pseudo compustat_annual", "domain": "Pseudo Data"},
    {"type": "compustat_quarterly", "dataset_name": "pseudo compustat_quarterly", "domain": "Pseudo Data"},
    {"type": "ccm_links", "dataset_name": "pseudo ccm_links", "domain": "Pseudo Data"},
]

# %% Supported other datasets
# List of dicts of the supported other datasets and their corresponding
# dataset names.
_OTHER_DATASETS = [
    {"type": "stock_prices", "dataset_name": "YahooFinance", "domain": "Stock Prices"},
    {"type": "constituents", "dataset_name": "various", "domain": "Index Constituents"},
    {"type": "fred", "dataset_name": "various", "domain": "FRED"},
    {"type": "osap", "dataset_name": "Open Source Asset Pricing", "domain": "Open Source Asset Pricing"},
    {"type": "risk_free", "dataset_name": "Risk-Free Rate", "domain": "Tidy Finance"},
    {"type": "high_frequency_sp500", "dataset_name": "High Frequency S&P 500", "domain": "Tidy Finance"},
    {"type": "factor_library", "dataset_name": "Factor Library", "domain": "Tidy Finance"},
    {"type": "factor_library_grid", "dataset_name": "Factor Library Grid", "domain": "Tidy Finance"},
]


# %% Public API


def list_supported_datasets(
    domain: Optional[str | list[str]] = None,
    as_vector: bool = False,
) -> "pd.DataFrame | list[str]":
    """List all datasets supported by ''download_data''.

    Aggregates the Global Q, Fama-French, Goyal-Welch, WRDS, Pseudo Data,
    and "other" tables into a single :class:'pandas.DataFrame'. The legacy
    Fama-French table is intentionally excluded from the master listing.

    Parameters
    ----------
    domain : str or list of str, optional
        Restrict the result to one or more domain labels (for example
        ''"WRDS"'' or ''["Fama-French", "Global Q"]'').
    as_vector : bool, default False
        If ''True'', return a list of dataset ''type'' strings instead of a
        DataFrame.

    Returns
    -------
    pandas.DataFrame or list of str
        Either a DataFrame with columns ''type'', ''dataset_name'', and
        ''domain'', or a list of ''type'' strings when ''as_vector=True''.
    """
    rows = (
        _Q_DATASETS
        + _FF_DATASETS
        + _MACRO_DATASETS
        + _WRDS_DATASETS
        + _PSEUDO_DATASETS
        + _OTHER_DATASETS
    )
    df = pd.DataFrame(rows)[["type", "dataset_name", "domain"]]

    if domain is not None:
        if isinstance(domain, str):
            filter_domains = [domain]
        else:
            filter_domains = list(domain)
        df = df[df["domain"].isin(filter_domains)].reset_index(drop=True)

    if as_vector:
        return df["type"].tolist()
    return df.reset_index(drop=True)


# %% Legacy-type translation helpers

# Domains that are valid both as a top-level domain name and as a "legacy"
# type string. When a value appears here it is treated as a domain, not
# a legacy type.
_SIMPLE_DOMAINS: tuple[str, ...] = (
    "constituents",
    "fred",
    "stock_prices",
    "osap",
)


# Canonical, human-readable domain names accepted by ``download_data``.
# These match the ``domain`` column returned by ``list_supported_datasets``.
_SUPPORTED_DOMAINS: tuple[str, ...] = (
    "Fama-French",
    "Global Q",
    "Goyal-Welch",
    "WRDS",
    "Pseudo Data",
    "Index Constituents",
    "FRED",
    "Stock Prices",
    "Open Source Asset Pricing",
    "Tidy Finance",
)


# Soft-deprecated machine-readable domain names mapped to their canonical
# human-readable replacements. Passing any of these to ``download_data``
# still works but emits a ``DeprecationWarning`` via
# ``_resolve_domain_alias``.
_DOMAIN_ALIASES: dict[str, str] = {
    "famafrench": "Fama-French",
    "factors_ff": "Fama-French",
    "globalq": "Global Q",
    "factors_q": "Global Q",
    "macro_predictors": "Goyal-Welch",
    "wrds": "WRDS",
    "pseudo": "Pseudo Data",
    "constituents": "Index Constituents",
    "fred": "FRED",
    "stock_prices": "Stock Prices",
    "osap": "Open Source Asset Pricing",
    "tidyfinance": "Tidy Finance",
}


def _resolve_domain_alias(domain: str) -> str:
    """Map a soft-deprecated domain alias to its canonical name.

    The machine-readable domain names used in earlier releases (for
    example ``"famafrench"``, ``"wrds"``, ``"pseudo"`` or
    ``"tidyfinance"``) are still accepted but now resolve to the
    canonical, human-readable names returned by
    ``list_supported_datasets``. Passing an alias emits a
    :class:`DeprecationWarning`. Any other value is returned unchanged.
    """
    canonical = _DOMAIN_ALIASES.get(domain)
    if canonical is not None:
        warnings.warn(
            f"The domain {domain!r} is deprecated; use {canonical!r} "
            "instead. See list_supported_datasets() for the canonical "
            "domain names.",
            DeprecationWarning,
            stacklevel=3,
        )
        return canonical
    return domain


def _parse_type_to_domain_dataset(
    type_str: str,
) -> tuple[str, Optional[str]]:
    """Translate a legacy ''type'' string into a ''(domain, dataset)'' pair.

    The dispatch rules are (domains are the canonical, human-readable
    names returned by ``list_supported_datasets``):

    * Fama-French legacy / current types resolve to
      ''("Fama-French", <dataset_name>)''.
    * Global Q types resolve to
      ''("Global Q", <dataset_name without trailing ".csv">)''.
    * ''macro_predictors_*'' strings resolve to
      ''("Goyal-Welch", <suffix>)''.
    * ''wrds_*'' strings resolve to ''("WRDS", <suffix>)''.
    * ''hf_*'' strings resolve to ''("Tidy Finance", <suffix>)''.
    * ''constituents'', ''fred'', ''stock_prices'', ''osap'' resolve to
      ''("Index Constituents", None)'', ''("FRED", None)'',
      ''("Stock Prices", None)'' and
      ''("Open Source Asset Pricing", None)'' respectively.
    * Anything else raises :class:'ValueError'.

    Parameters
    ----------
    type_str : str
        The legacy ''type'' string passed by the caller.

    Returns
    -------
    tuple of (str, str or None)
        The resolved ''(domain, dataset)'' pair.

    Raises
    ------
    ValueError
        If ''type_str'' does not match any known legacy pattern.
    """
    # Fama-French (current + legacy share the lookup table)
    for row in _FF_DATASETS:
        if row["type"] == type_str:
            return ("Fama-French", row["dataset_name"])
    for row in _FF_LEGACY_DATASETS:
        if row["type"] == type_str:
            return ("Fama-French", row["dataset_name"])

    # Global Q
    for row in _Q_DATASETS:
        if row["type"] == type_str:
            ds = row["dataset_name"]
            if ds.endswith(".csv"):
                ds = ds[: -len(".csv")]
            return ("Global Q", ds)

    # Macro predictors
    for row in _MACRO_DATASETS:
        if row["type"] == type_str:
            suffix = (
                type_str[len("macro_predictors_"):]
                if type_str.startswith("macro_predictors_")
                else type_str
            )
            return ("Goyal-Welch", suffix)

    # WRDS
    for row in _WRDS_DATASETS:
        if row["type"] == type_str:
            suffix = (
                type_str[len("wrds_"):]
                if type_str.startswith("wrds_")
                else type_str
            )
            return ("WRDS", suffix)

    # High-frequency datasets hosted on Hugging Face
    if type_str.startswith("hf_"):
        return ("Tidy Finance", type_str[len("hf_"):])

    # Simple domain-only datasets
    if type_str in _SIMPLE_DOMAINS:
        return (_DOMAIN_ALIASES[type_str], None)

    raise ValueError(
        f"Cannot parse legacy type: {type_str!r}. "
        "Use list_supported_datasets() to see available datasets."
    )


def _is_legacy_type(x: str) -> bool:
    """Return ''True'' iff ''x'' is a legacy ''type'' string.

    A value is considered legacy when ''_parse_type_to_domain_dataset''
    would succeed on it *and* it is not one of the simple domain names
    listed in :data:'_SIMPLE_DOMAINS' (those are already valid domains in
    their own right).  ''Tidy Finance''-domain "other" datasets such as
    ''risk_free'' or ''factor_library'' are not treated as legacy either.
    """
    if x in _SIMPLE_DOMAINS:
        return False

    known_types: set[str] = set()
    known_types.update(row["type"] for row in _FF_DATASETS)
    known_types.update(row["type"] for row in _FF_LEGACY_DATASETS)
    known_types.update(row["type"] for row in _Q_DATASETS)
    known_types.update(row["type"] for row in _MACRO_DATASETS)
    known_types.update(row["type"] for row in _WRDS_DATASETS)
    # "other" rows that are NOT in the tidyfinance domain and NOT "osap"
    known_types.update(
        row["type"]
        for row in _OTHER_DATASETS
        if row["domain"] != "Tidy Finance" and row["type"] != "osap"
    )

    return x in known_types


def _check_supported_domain(domain: str) -> None:
    """Raise :class:'ValueError' when ''domain'' is not supported.

    The list of supported domains is exposed via :data:'_SUPPORTED_DOMAINS'.
    """
    if domain not in _SUPPORTED_DOMAINS:
        joined = ", ".join(repr(d) for d in _SUPPORTED_DOMAINS)
        raise ValueError(
            f"Unsupported domain: {domain!r}. Supported domains: {joined}."
        )


def _is_legacy_type_wrds(x: str) -> bool:
    """Return True if x is a legacy WRDS type string (starts with 'wrds_')."""
    return isinstance(x, str) and x.startswith("wrds_")


_WRDS_SUPPORTED_DATASETS = (
    "crsp_monthly",
    "crsp_daily",
    "compustat_annual",
    "compustat_quarterly",
    "ccm_links",
    "fisd",
    "trace_enhanced",
)


def _check_supported_dataset_wrds(dataset: str) -> None:
    """Raise ValueError if dataset is not a supported WRDS dataset."""
    if dataset not in _WRDS_SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported WRDS dataset: {dataset!r}. "
            f"Supported: {', '.join(_WRDS_SUPPORTED_DATASETS)}."
        )


_WRDS_CRSP_SUPPORTED_DATASETS = ("crsp_monthly", "crsp_daily")


def _check_supported_dataset_wrds_crsp(dataset: str) -> None:
    """Raise ValueError if dataset is not a supported CRSP dataset."""
    if dataset not in _WRDS_CRSP_SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported CRSP dataset: {dataset!r}. "
            f"Supported: {', '.join(_WRDS_CRSP_SUPPORTED_DATASETS)}."
        )


def _is_legacy_type_ff(x: str) -> bool:
    """Return True if x is a known Fama-French type (current or legacy)."""
    types = {row["type"] for row in _FF_DATASETS}
    types.update(row["type"] for row in _FF_LEGACY_DATASETS)
    return x in types


def _is_legacy_type_q(x: str) -> bool:
    """Return True if x is a known Global Q dataset type."""
    return x in {row["type"] for row in _Q_DATASETS}


def _determine_frequency_ff(dataset: str) -> str:
    """Map a Fama-French dataset name to its reporting frequency."""
    if "[Daily]" in dataset:
        return "daily"
    if "[Weekly]" in dataset:
        return "weekly"
    return "monthly"


def _is_breakpoints_ff(dataset: str) -> bool:
    """Return True if dataset is a Fama-French breakpoints file."""
    return "Breakpoints" in dataset


def _determine_frequency_q(dataset: str) -> str:
    """Map a Global Q dataset name to its reporting frequency."""
    lowered = dataset.lower()
    for frequency in ("daily", "weekly", "monthly", "quarterly", "annual"):
        if frequency in lowered:
            return frequency
    raise ValueError(
        f"Cannot determine frequency from dataset name: {dataset!r}."
    )


def _check_supported_dataset_ff(dataset: str) -> str:
    """Validate a Fama-French dataset_name and return its source file URL.

    Raises
    ------
    ValueError
        If dataset is not a supported Fama-French dataset_name.
    """
    for row in _FF_DATASETS + _FF_LEGACY_DATASETS:
        if row["dataset_name"] == dataset:
            return row["file_url"]
    raise ValueError(
        f"Unsupported Fama-French dataset: {dataset!r}. "
        "Use list_supported_datasets(domain='Fama-French') to see "
        "available datasets."
    )


def _check_supported_dataset_q(dataset: str) -> None:
    """Raise ValueError if dataset is not a supported Global Q dataset_name."""
    if dataset not in {row["dataset_name"] for row in _Q_DATASETS}:
        raise ValueError(
            f"Unsupported Global Q dataset: {dataset!r}. "
            "Use list_supported_datasets(domain='Global Q') to see "
            "available datasets."
        )
