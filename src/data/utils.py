import numpy as np
import polars as pl


def split_history_horizon(
    df: pl.DataFrame,
    history: int,
    horizon: int,
    tgt_cols: list | None = None,
    time_col: str | None = None,
):
    if tgt_cols is None:
        tgt_cols = [c for c in df.columns if c != time_col]
        ctx_cols = []
    else:
        ctx_cols = [c for c in df.columns if c not in tgt_cols]

    if time_col:
        if df.schema[time_col].dtype == pl.Datetime:
            time_expr = pl.col(time_col).dt.timestamp()
        else:
            time_expr = pl.col(time_col).cast(float)
    else:
        time_expr = pl.col("index").cast(float)

    df_idx = df.with_row_index()

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

    ctx = np.stack([ctx_df[c].to_numpy() for c in ctx_df.columns], axis=-1)
    obs = np.stack([obs_df[c].to_numpy() for c in tgt_cols], axis=-1)
    tgt = np.stack([tgt_df[c].to_numpy() for c in tgt_cols], axis=-1)

    return ctx, obs, tgt


def standard_scale(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame):
    cols = train_df.columns
    mean = train_df.select(pl.all().mean())
    std = train_df.select(pl.all().std())
    train_df = train_df.select([pl.col(c) - mean[c] / std[c] for c in cols])
    val_df = val_df.select([pl.col(c) - mean[c] / std[c] for c in cols])
    test_df = test_df.select([pl.col(c) - mean[c] / std[c] for c in cols])
    return train_df, val_df, test_df
