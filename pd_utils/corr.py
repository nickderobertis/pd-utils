from typing import Sequence

import pandas as pd
import numpy as np


def formatted_corr_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    corr_df = df[cols].corr()
    corr_df = _lower_triangular_of_df(corr_df)
    return corr_df.applymap(lambda x: f'{x:.2f}' if not isinstance(x, str) else x)


def _lower_triangular_of_df(df):
    return pd.DataFrame(np.tril(df), index=df.index, columns=df.columns).replace(0, '')
