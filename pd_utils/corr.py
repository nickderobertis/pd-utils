from typing import Sequence, Optional

import pandas as pd
import numpy as np


def formatted_corr_df(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Calculates correlations on a DataFrame and displays only the lower triangular of the
    resulting correlation DataFrame.

    :param df:
    :param cols: subset of column names on which to calculate correlations
    :return:
    """
    if not cols:
        use_cols = list(df.columns)
    else:
        use_cols = list(cols)

    corr_df = df[use_cols].corr()
    corr_df = _lower_triangular_of_df(corr_df)
    return corr_df.applymap(lambda x: f'{x:.2f}' if not isinstance(x, str) else x)


def _lower_triangular_of_df(df):
    return pd.DataFrame(np.tril(df), index=df.index, columns=df.columns).replace(0, '')
