import pandas as pd

from dero.ext_pandas.optimize.typing import DfOrSeries


def read_file(filepath: str, **read_func_kwargs) -> DfOrSeries:
    extension = filepath.rpartition('.')[-1].lower()
    if extension == 'csv':
        df = pd.read_csv(filepath, **read_func_kwargs)
    # TODO: determine filetype and use proper loader
    else:
        raise NotImplementedError(f'could not load filetype {extension}')

    return df