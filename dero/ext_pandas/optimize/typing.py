from typing import Union, Tuple, Dict

import pandas as pd

DfOrSeries = Union[pd.DataFrame, pd.Series]
PdDTypeQuadTuple = Tuple[DfOrSeries, DfOrSeries, DfOrSeries, DfOrSeries]
StrDict = Dict[str, str]