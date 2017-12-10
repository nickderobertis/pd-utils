import statsmodels.api as sm
import pandas as pd
from numpy import nan

from .pdutils import split

def reg_by(df, yvar, xvars, groupvar, merge=False, cons=True):
    """
    Runs a regression of df[yvar] on df[xvars] by values of groupvar. Outputs a dataframe with values of
    groupvar and corresponding coefficients, unless merge=True, then outputs the original dataframe with the
    appropriate coefficients merged in.

    Required inputs:
    groupvar: str or list of strs, column names of columns identifying by groups

    Optional Options:
    cons: True to include a constant, False to not
    """
    result_df = pd.DataFrame()

    if isinstance(xvars, str):
        xvars = [xvars]
    assert isinstance(xvars, list)

    drop_group = False
    if isinstance(groupvar, list):
        df['__key_regby__'] = ''
        for var in groupvar:
            df['__key_regby__'] = df['__key_regby__'] + df[var].astype(str)
        groupvar = '__key_regby__'
        drop_group = True

    # Select dataframe of only y and x vars
    yx_df = df.loc[:, xvars + [yvar]]
    # Recombine groupvar and drop missing
    yx_df = pd.concat([yx_df, df[groupvar]], axis=1).dropna()

    if cons:
        rhs = ['const'] + xvars
    else:
        rhs = xvars

    arrs = split(yx_df, [yvar] + xvars, keyvar=groupvar)
    groups = df[groupvar].unique().tolist()
    assert len(arrs) == len(groups)

    for i, arr in enumerate(arrs):
        X = arr[:, 1:].astype(float)

        if cons:
            X = sm.add_constant(X)

        y = arr[:, 0].astype(float)

        if arr.shape[0] > len(xvars) + 1:  # if enough observations, run regression
            model = sm.OLS(y, X)
            result = model.fit()
            this_result = pd.DataFrame(result.params).T
        else:  # not enough obs, return nans
            this_result = pd.DataFrame(data=[nan for i in range(len(rhs))]).T

        this_result[groupvar] = groups[i]
        result_df = result_df.append(this_result)  # Or whatever summary info you want

    cols = rhs + [groupvar]
    result_df.columns = ['coef_' + col if col not in (groupvar, 'const') else col for col in cols]

    if merge:
        result_df = df.merge(result_df, how='left', on=groupvar)
    if drop_group:
        result_df.drop(groupvar, axis=1, inplace=True)

    return result_df.reset_index(drop=True)