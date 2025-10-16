"""File with some data-related utilities."""

import polars as pl
import torch
from torch import Tensor

from .date_utils import temporal_cyclic_features


def process_split(
    df: pl.DataFrame,
    context: int,
    horizon: int,
    time_col: str,
    has_time: bool,
    has_date: bool,
    target_cols: list,
    target_mean: Tensor,
    target_scale: Tensor,
    dt_mean: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    time_feat_exprs = [
        (pl.col(time_col).diff() / dt_mean).fill_null(0).alias("dt"),
        pl.col("time_norm"),
    ]

    if has_time:
        time_feat_exprs.extend(temporal_cyclic_features(pl.col(time_col), "1d"))

    if has_date:
        for period in ["1w", "1m", "1q", "1y"]:
            time_feat_exprs.extend(temporal_cyclic_features(pl.col(time_col), period))

    df_idx = df.with_row_index()

    t_df = (
        df_idx.select("index", *time_feat_exprs)
        .rolling("index", period=f"{context + horizon}i", offset=f"-{context}i")
        .agg(pl.all())
        .slice(context, -horizon)
        .drop("index")
        .cast(pl.Array(pl.Float32, context + horizon))
    )

    x_df = (
        df_idx.rolling("index", period=f"{context}i")
        .agg(*target_cols)
        .slice(context, -horizon)
        .select(pl.col(*target_cols).list.to_array(context))
        .cast(pl.Array(pl.Float32, context))
    )

    y_df = (
        df_idx.rolling("index", period=f"{horizon}i", offset="0i")
        .agg(*target_cols)
        .slice(context, -horizon)
        .select(pl.col(*target_cols).list.to_array(horizon))
        .cast(pl.Array(pl.Float32, horizon))
    )

    t = torch.stack([t_df[c].to_torch() for c in t_df.columns], axis=-1)
    x = torch.stack([x_df[c].to_torch() for c in target_cols], axis=-1)
    y = torch.stack([y_df[c].to_torch() for c in target_cols], axis=-1)

    x = (x - target_mean) / target_scale
    y = (y - target_mean) / target_scale

    return t, x, y
