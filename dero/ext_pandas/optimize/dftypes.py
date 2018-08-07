import pandas as pd

from dero.ext_pandas.optimize import PdDTypeQuadTuple


def optimized_df(df: pd.DataFrame) -> pd.DataFrame:
    columns = [col for col in df.columns]

    type_dfs: PdDTypeQuadTuple = _optimized_dtype_dfs(df)
    return pd.concat(type_dfs, axis=1)[columns]


def _optimized_dtype_dfs(df: pd.DataFrame) -> PdDTypeQuadTuple:
    obj_df = df.select_dtypes(include=['object'])
    obj_df = obj_df.astype('category')

    int_df = df.select_dtypes(include=['int'])
    int_df = int_df.apply(pd.to_numeric, downcast='unsigned')

    float_df = df.select_dtypes(include=['float'])
    float_df = float_df.apply(pd.to_numeric, downcast='float')

    type_dfs = (obj_df, int_df, float_df)

    optimized_columns = []
    for type_df in type_dfs:
        optimized_columns += [col for col in type_df.columns]

    # Excluded dtype should just be 'datetime', which does not need conversion
    excluded_columns = [col for col in df.columns if col not in optimized_columns]

    return type_dfs + (df[excluded_columns],)