import pandas as pd

from pd_utils.optimize.typing import DfOrSeries


def read_file(filepath: str, **read_func_kwargs) -> DfOrSeries:
    extension = filepath.rpartition('.')[-1].lower()
    if extension == 'csv':
        df = pd.read_csv(filepath, **read_func_kwargs)
    # TODO: determine filetype and use proper loader
    else:
        raise NotImplementedError(f'could not load filetype {extension}')
    #### TEMP
    # from pd_utils.optimize.dftypes import optimized_df
    # df = optimized_df(df)
    ### END TEMP

    return df