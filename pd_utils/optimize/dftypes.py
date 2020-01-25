from typing import List, Union

import pandas as pd
from pd_utils.optimize.typing import PdDTypeQuadTuple, StrDict


def optimized_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in df.columns]

    type_dfs: PdDTypeQuadTuple = _optimized_dtype_dfs(df)
    return pd.concat(type_dfs, axis=1)[columns]


def _optimized_dtype_dfs(df: pd.DataFrame) -> PdDTypeQuadTuple:
    obj_df = df.select_dtypes(include=["object"])
    if not obj_df.empty:
        obj_df = obj_df.astype("category")

    int_df = df.select_dtypes(include=["int"])
    if not int_df.empty:
        int_df = int_df.apply(pd.to_numeric, downcast="unsigned")

    float_df = df.select_dtypes(include=["float"])
    if not float_df.empty:
        float_df = float_df.apply(pd.to_numeric, downcast="float")

    type_dfs = (obj_df, int_df, float_df)

    optimized_columns: List[Union[str, float, int]] = []
    for type_df in type_dfs:
        optimized_columns += [col for col in type_df.columns]

    # Excluded dtype should just be 'datetime', which does not need conversion
    excluded_columns = [col for col in df.columns if col not in optimized_columns]

    return type_dfs + (df[excluded_columns],)


def df_types_dict(df: pd.DataFrame, remove_dates=True) -> StrDict:
    df_types_dict = _df_types_dict(df)
    if remove_dates:
        return {
            col_name: dtype
            for col_name, dtype in df_types_dict.items()
            if "date" not in dtype
        }
    else:
        return df_types_dict


def _df_types_dict(df: pd.DataFrame) -> StrDict:
    return df.dtypes.apply(lambda x: str(x)).to_dict()
