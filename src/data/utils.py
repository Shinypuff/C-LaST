"""File with some data-related utilities."""

import polars as pl
import torch
from torch import Tensor


def split_history_horizon(
    df: pl.DataFrame,
    history: int,
    horizon: int,
    tgt_cols: list | None = None,
    time_col: str | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split a DataFrame into context, observation, and target tensors based on history and horizon windows.

    This function processes a DataFrame to create sequences for time series modeling. It uses rolling windows
    to extract context (input) sequences, observation (target input) sequences, and target (prediction) sequences.
    The function handles both target columns and context columns, and optionally uses a time column for temporal information.

    Args:
        df (pl.DataFrame): The input DataFrame containing time series data.
        history (int): The length of the history window (number of past time steps to consider).
        horizon (int): The length of the horizon window (number of future time steps to predict).
        tgt_cols (list | None, optional): List of column names to be used as target variables.
                                        If None, all columns except the time column are used. Defaults to None.
        time_col (str | None, optional): The name of the time column. If provided, it is used for temporal operations.
                                        Defaults to None.

    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - ctx (Tensor): Context tensor with shape (batch, history + horizon, num_ctx_cols).
            - obs (Tensor): Observation tensor with shape (batch, history, num_tgt_cols).
            - tgt (Tensor): Target tensor with shape (batch, horizon, num_tgt_cols).

    """
    if tgt_cols is None:
        tgt_cols = [c for c in df.columns if c != time_col]
        ctx_cols = []
    else:
        ctx_cols = [c for c in df.columns if c not in tgt_cols and c != time_col]

    if time_col:
        if df.schema[time_col] == pl.Datetime:
            time_expr = pl.col(time_col).dt.timestamp()
        else:
            time_expr = pl.col(time_col)
    else:
        time_expr = pl.col("index")

    df_idx = df.cast(pl.Float32).with_row_index()

    ctx_df = (
        df_idx.rolling("index", period=f"{history + horizon}i", offset=f"-{history}i")
        .agg(time_expr.diff().fill_null(0), *ctx_cols)
        .slice(history, -horizon)
        .select(pl.exclude("index").list.to_array(history + horizon))
    )

    obs_df = (
        df_idx.rolling("index", period=f"{history}i")
        .agg(*tgt_cols)
        .slice(history, -horizon)
        .select(pl.col(*tgt_cols).list.to_array(history))
    )

    tgt_df = (
        df_idx.rolling("index", period=f"{horizon}i", offset="0i")
        .agg(*tgt_cols)
        .slice(history, -horizon)
        .select(pl.col(*tgt_cols).list.to_array(horizon))
    )

    ctx = torch.stack([ctx_df[c].to_torch() for c in ctx_df.columns], axis=-1)
    obs = torch.stack([obs_df[c].to_torch() for c in tgt_cols], axis=-1)
    tgt = torch.stack([tgt_df[c].to_torch() for c in tgt_cols], axis=-1)

    return ctx, obs, tgt
