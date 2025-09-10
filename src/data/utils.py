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


def standard_scale(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame):
    cols = train_df.columns
    mean = train_df.select(pl.all().mean())
    std = train_df.select(pl.all().std())
    train_df = train_df.select([pl.col(c) - mean[c] / std[c] for c in cols])
    val_df = val_df.select([pl.col(c) - mean[c] / std[c] for c in cols])
    test_df = test_df.select([pl.col(c) - mean[c] / std[c] for c in cols])
    return train_df, val_df, test_df
