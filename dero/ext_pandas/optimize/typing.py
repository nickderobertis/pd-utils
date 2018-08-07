from typing import Union, Tuple

import pandas as pd

DfOrSeries = Union[pd.DataFrame, pd.Series]
PdDTypeQuadTuple = Tuple[DfOrSeries, DfOrSeries, DfOrSeries, DfOrSeries]