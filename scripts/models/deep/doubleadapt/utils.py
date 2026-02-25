"""
Data utilities for DoubleAdapt + FiLM.

Provides rolling task organization, data extraction, and preprocessing
with dual-path stock + macro data handling.

Based on DoubleAdapt's utils.py with macro feature extension.
"""

import bisect
import traceback
import warnings
from typing import Union, List, Tuple, Dict, Text

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class TimeAdjuster:
    """Find appropriate trade dates and align date segments to calendar."""

    def __init__(self, calendar):
        if isinstance(calendar, list):
            calendar = pd.Series(data=calendar)
        self.cals = pd.to_datetime(calendar)

    def get(self, idx):
        if idx is None or idx >= len(self.cals):
            return None
        return self.cals[idx]

    def max(self):
        return max(self.cals)

    def align_idx(self, time_point, tp_type="start"):
        if time_point is None:
            return None
        time_point = pd.Timestamp(time_point)
        if tp_type == "start":
            return bisect.bisect_left(self.cals, time_point)
        elif tp_type == "end":
            return bisect.bisect_right(self.cals, time_point) - 1
        raise NotImplementedError(f"Unsupported tp_type: {tp_type}")

    def align_time(self, time_point, tp_type="start"):
        if time_point is None:
            return None
        return self.cals[self.align_idx(time_point, tp_type=tp_type)]

    def align_seg(self, segment):
        if isinstance(segment, dict):
            return {k: self.align_seg(seg) for k, seg in segment.items()}
        elif isinstance(segment, (tuple, list)):
            return (
                self.align_time(segment[0], tp_type="start"),
                self.align_time(segment[1], tp_type="end"),
            )
        raise NotImplementedError(f"Unsupported segment type: {type(segment)}")

    SHIFT_SD = "sliding"
    SHIFT_EX = "expanding"

    def shift(self, seg, step, rtype=SHIFT_SD):
        if isinstance(seg, tuple):
            start_idx = self.align_idx(seg[0], tp_type="start")
            end_idx = self.align_idx(seg[1], tp_type="end")
            if rtype == self.SHIFT_SD:
                start_idx = start_idx + step if start_idx is not None else None
                end_idx = end_idx + step if end_idx is not None else None
            elif rtype == self.SHIFT_EX:
                end_idx = end_idx + step if end_idx is not None else None
            else:
                raise NotImplementedError(f"Unsupported rtype: {rtype}")
            if start_idx is not None and start_idx > len(self.cals):
                raise KeyError("Segment is out of valid calendar")
            return self.get(start_idx), self.get(end_idx)
        raise NotImplementedError(f"Unsupported seg type: {type(seg)}")


# ============================================================
# Rolling task organization
# ============================================================

def organize_all_tasks(segments, ta, step, trunc_days=2,
                       rtype=TimeAdjuster.SHIFT_SD):
    """Organize training/validation/test data into rolling tasks.

    Args:
        segments: dict with 'train', 'valid', 'test' date tuples
        ta: TimeAdjuster
        step: rolling interval in trading days
        trunc_days: gap between train end and test start (avoid leakage)
        rtype: 'sliding' or 'expanding' window
    Returns:
        dict with 'train', 'valid', 'test' lists of task segments
    """
    all_tasks = _organize_tasks(
        segments['train'][0], segments['test'][-1],
        ta, step, trunc_days, rtype,
    )
    rolling_tasks = {}
    rolling_tasks['train'], rolling_tasks['test'] = _split_tasks(
        all_tasks, split_point=segments['valid'][-1]
    )
    rolling_tasks['train'], rolling_tasks['valid'] = _split_tasks(
        rolling_tasks['train'], split_point=segments['train'][-1]
    )
    return rolling_tasks


def _split_tasks(rolling_tasks, split_point):
    assert len(rolling_tasks) > 0
    for i, t in enumerate(rolling_tasks):
        if t["test"][-1] >= pd.Timestamp(split_point):
            break
    return rolling_tasks[:i], rolling_tasks[i:]


def _organize_tasks(start_date, end_date, ta, step, trunc_days=2,
                    rtype=TimeAdjuster.SHIFT_SD):
    train_begin = start_date
    train_end = ta.get(ta.align_idx(train_begin) + step - 1)
    test_begin = ta.get(ta.align_idx(train_begin) + step - 1 + trunc_days)
    test_end = ta.get(ta.align_idx(test_begin) + step - 1)
    first_task = {
        "train": (train_begin, train_end),
        "test": (test_begin, test_end),
    }
    return _generate_rolling(first_task, step, ta, end_date, rtype)


def _generate_rolling(first_task, step, ta, end_date, rtype):
    rolling_tasks = [first_task]
    while True:
        try:
            task = {}
            for k, v in rolling_tasks[-1].items():
                if k == 'train' and rtype == TimeAdjuster.SHIFT_EX:
                    task[k] = ta.shift(v, step=step, rtype=rtype)
                else:
                    task[k] = ta.shift(v, step=step, rtype=TimeAdjuster.SHIFT_SD)
            rolling_tasks.append(task)
            if rolling_tasks[-1]['test'][-1] >= ta.align_time(end_date, tp_type='end'):
                rolling_tasks[-1]['test'] = (
                    rolling_tasks[-1]['test'][0],
                    ta.align_time(end_date, tp_type='end'),
                )
                break
        except Exception:
            traceback.print_exc()
            break
    return rolling_tasks


# ============================================================
# Data extraction with macro
# ============================================================

def get_rolling_data_with_macro(
    rolling_task_segments,
    stock_data,
    macro_df,
    macro_cols,
    factor_num=5,
    horizon=1,
    sequence_last_dim=True,
    macro_lag=1,
):
    """Extract rolling task data with both stock and macro features.

    Args:
        rolling_task_segments: list of task segment dicts from organize_all_tasks
        stock_data: DataFrame with MultiIndex (datetime, instrument), feature + label columns
        macro_df: DataFrame of macro features (date-indexed)
        macro_cols: list of macro column names
        factor_num: features per timestep (5 for OHLCV)
        horizon: prediction horizon
        sequence_last_dim: if True, stock data is [batch, factor_num, seq_len]
        macro_lag: days to lag macro data
    Returns:
        list of task dicts with stock tensors, macro tensors, labels, and metadata
    """
    from models.common.macro_features import prepare_macro

    rolling_tasks_data = []
    for seg in tqdm(rolling_task_segments, desc="creating tasks"):
        task = _get_task_data(seg, stock_data)
        if task is None:
            continue
        rolling_tasks_data.append(task)

    return _preprocess_with_macro(
        rolling_tasks_data,
        macro_df=macro_df,
        macro_cols=macro_cols,
        factor_num=factor_num,
        sequence_last_dim=sequence_last_dim,
        H=1 + horizon,
        macro_lag=macro_lag,
    )


def _get_task_data(segments, dataframe):
    """Extract raw data for a single rolling task."""
    train_exist = "train" in segments
    test_segs = [str(dt) for dt in segments["test"]]
    d_test = _get_data_from_seg(test_segs, dataframe, test=True)
    if d_test is None:
        return None

    result = dict(
        X_test=d_test["feature"],
        y_test=d_test["label"].iloc[:, 0],
        test_idx=d_test["label"].index,
    )
    if train_exist:
        train_segs = [str(dt) for dt in segments["train"]]
        d_train = _get_data_from_seg(train_segs, dataframe)
        if d_train is not None:
            result.update(
                X_train=d_train["feature"],
                y_train=d_train["label"].iloc[:, 0],
                train_idx=d_train["label"].index,
            )
    return result


def _get_data_from_seg(seg, dataframe, test=False):
    """Extract data from a date segment of the dataframe."""
    try:
        d = (
            dataframe.loc(axis=0)[seg[0]: seg[1]]
            if not test or seg[1] <= str(dataframe.index[-1][0])
            else dataframe.loc(axis=0)[seg[0]:]
        )
    except Exception:
        traceback.print_exc()
        all_dates = dataframe.index.levels[0]
        new_seg = [seg[0], seg[1]]
        if seg[0] not in all_dates:
            candidates = all_dates[all_dates > seg[0]]
            if len(candidates) == 0:
                return None
            new_seg[0] = candidates[0]
            if str(new_seg[0])[:10] > seg[1]:
                warnings.warn(f"Exceed test time {new_seg}")
                return None
        if seg[1] not in all_dates:
            candidates = all_dates[all_dates < seg[1]]
            if len(candidates) == 0:
                return None
            new_seg[1] = candidates[-1]
            if str(new_seg[1])[:10] < seg[0]:
                warnings.warn(f"Exceed training time {new_seg}")
                return None
        d = (
            dataframe.loc(axis=0)[new_seg[0]: new_seg[1]]
            if not test or new_seg[1] <= all_dates[-1]
            else dataframe.loc(axis=0)[new_seg[0]:]
        )
        warnings.warn(f"{seg} becomes {new_seg} after adjustment")
    return d


def _preprocess_with_macro(
    task_data_list,
    macro_df,
    macro_cols,
    factor_num=5,
    sequence_last_dim=True,
    H=1,
    macro_lag=1,
):
    """Preprocess task data: reshape stock features, prepare macro tensors.

    Stock: reshape to [batch, seq_len, factor_num] for FeatureAdapter
    Macro: [batch, n_macro] aligned to stock dates (no reshape)
    """
    from models.common.macro_features import prepare_macro

    skip_ids = []
    for i, task_data in enumerate(task_data_list):
        data_type = set()
        for k in list(task_data.keys()):
            if k.startswith("X") or k.startswith("y"):
                dt = k[2:]
                data_type.add(dt)

        # Prepare macro for each split
        for dt in data_type:
            idx_key = dt + "_idx" if dt != "" else "idx"
            # Determine index for macro alignment
            if dt == "train" and "train_idx" in task_data:
                idx = task_data["train_idx"]
            elif dt == "test" and "test_idx" in task_data:
                idx = task_data["test_idx"]
            else:
                continue

            macro_vals = prepare_macro(idx, macro_df, macro_cols, macro_lag)
            task_data[f"macro_{dt}"] = torch.tensor(macro_vals, dtype=torch.float32)

        # Convert stock features and labels to tensors
        for k in list(task_data.keys()):
            if k.startswith("X") or k.startswith("y"):
                if not isinstance(task_data[k], np.ndarray):
                    task_data[k] = task_data[k].to_numpy()
                task_data[k] = torch.tensor(task_data[k], dtype=torch.float32)

        if task_data['y_test'].shape[0] == 0:
            skip_ids.append(i)

        # Reshape stock features: [batch, factor_num*seq_len] -> [batch, seq_len, factor_num]
        for dt in data_type:
            k = "X_" + dt
            if k not in task_data:
                continue
            if sequence_last_dim:
                # alpha300 format: [batch, 300] -> [batch, 5, 60] -> permute -> [batch, 60, 5]
                task_data[k] = task_data[k].reshape(len(task_data[k]), factor_num, -1)
                task_data[k] = task_data[k].permute(0, 2, 1)
            else:
                task_data[k] = task_data[k].reshape(len(task_data[k]), -1, factor_num)

        # Compute meta_end: valid prediction window considering horizon
        test_date = task_data["test_idx"].codes[0] - task_data["test_idx"].codes[0][0]
        task_data["meta_end"] = (test_date <= (test_date[-1] - H + 1)).sum()

    # Handle tasks with empty test data
    if skip_ids:
        i = 0
        while i < len(skip_ids):
            task_w_train = task_data_list[skip_ids[i]]
            while i + 1 < len(skip_ids) and skip_ids[i + 1] == skip_ids[i] + 1:
                i += 1
            not_skip = skip_ids[i] + 1
            if not_skip < len(task_data_list):
                for key in task_w_train.keys():
                    if 'train' in key:
                        task_data_list[not_skip][key] = task_w_train[key]
            i += 1
        task_data_list = [
            task_data_list[j] for j in range(len(task_data_list)) if j not in skip_ids
        ]

    return task_data_list
