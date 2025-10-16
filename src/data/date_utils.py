from math import pi

import polars as pl


def temporal_cyclic_features(timecol: pl.Expr, period: str):
    start = timecol.dt.truncate(period)
    end = start.dt.offset_by(period)
    normalized = (timecol - start) / (end - start)
    cos = (normalized * 2 * pi).cos().alias(f"{period}_cos")
    sin = (normalized * 2 * pi).sin().alias(f"{period}_sin")
    return cos, sin
