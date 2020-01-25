import pandas as pd

from pd_utils.optimize.typing import DfOrSeries


def read_file(filepath: str, **read_func_kwargs) -> DfOrSeries:
    extension = filepath.rpartition(".")[-1].lower()
    if extension == "csv":
        df = pd.read_csv(filepath, **read_func_kwargs)
    # TODO [#4]: in read_file, determine filetype and use proper loader
    #
    # Currently it just handles csv
    else:
        raise NotImplementedError(f"could not load filetype {extension}")
    # TODO [#5]: make optimize df an option for read_file

    #### TEMP
    # from pd_utils.optimize.dftypes import optimized_df
    # df = optimized_df(df)
    ### END TEMP

    return df
