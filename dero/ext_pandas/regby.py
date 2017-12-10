import statsmodels.api as sm
import pandas as pd
import numpy as np
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
    # Convert xvars to list if str is passed
    xvars = _check_inputs_regby(xvars)

    # If there are multiple groupvars, create a key variable containing all groupvars (modifies df inplace)
    groupvar, drop_group = _set_groupvar_and_drop_group(df, groupvar)

    # Select dataframe of only y, x and group vars and drop any missings
    yx_df = df_for_reg(df, yvar, xvars, groupvar)

    # Create a list of right hand side variables. Includes 'const' if cons is True
    rhs = _set_rhs(xvars, cons)

    # Split DataFrame into a list of arrays with each bygroup being one array. Provide an accompanying list of bygroups
    arrs, groups = _get_lists_of_arrays_and_groups(yx_df, yvar, xvars, groupvar)

    # Run regressions by groups, storing results as a list of numpy arrays
    results = _reg_by(arrs, groups, xvars, rhs, cons)

    # Combine list of arrays into df, and apply column labels
    result_df = _result_list_of_arrays_to_df(results, rhs, groupvar)

    if merge:
        result_df = df.merge(result_df, how='left', on=groupvar)
    if drop_group:
        result_df.drop(groupvar, axis=1, inplace=True)

    return result_df.reset_index(drop=True)

def df_for_reg(df, yvar, xvars, groupvar):
    # Select dataframe of only y and x vars
    yx_df = df.loc[:, xvars + [yvar]]
    # Recombine groupvar and drop missing
    yx_df = pd.concat([yx_df, df[groupvar]], axis=1).dropna()

    return yx_df

def _reg_by(arrs, groups, xvars, rhs, cons):
    results = []
    for i, arr in enumerate(arrs):
        results.append(_reg(arr, xvars, rhs, cons, groups[i]))
    return results

def _reg(arr, xvars, rhs, cons, group):
    X = arr[:, 1:].astype(float)

    if cons:
        X = sm.add_constant(X)

    y = arr[:, 0].astype(float)

    if arr.shape[0] > len(xvars) + 1:  # if enough observations, run regression
        model = sm.OLS(y, X)
        result = model.fit()
        this_result = np.append(result.params, group)  # add groupvar
        this_result = this_result[None, :]  # cast 1d array into 2d array
    else:  # not enough obs, return nans
        this_result = np.empty((1, len(rhs) + 1), dtype='O')
        this_result[:] = nan
        this_result[0, len(rhs)] = group

    return this_result


def _check_inputs_regby(xvars):
    if isinstance(xvars, str):
        xvars = [xvars]
    assert isinstance(xvars, list)

    return xvars

def _set_groupvar_and_drop_group(df, groupvar):
    drop_group = False
    if isinstance(groupvar, list):
        df['__key_regby__'] = ''
        for var in groupvar:
            df['__key_regby__'] = df['__key_regby__'] + df[var].astype(str)
        groupvar = '__key_regby__'
        drop_group = True

    return groupvar, drop_group

def _set_rhs(xvars, cons):
    if cons:
        rhs = ['const'] + xvars
    else:
        rhs = xvars

    return rhs

def _result_list_of_arrays_to_df(results, rhs, groupvar):
    result_df = pd.DataFrame(np.concatenate(results, axis=0))
    result_df = result_df.apply(pd.to_numeric, errors='ignore')
    cols = rhs + [groupvar]
    result_df.columns = ['coef_' + col if col not in (groupvar, 'const') else col for col in cols]

    return result_df

def _get_lists_of_arrays_and_groups(df, yvar, xvars, groupvar):
    arrs = split(df, [yvar] + xvars, keyvar=groupvar)
    groups = df[groupvar].unique().tolist()
    assert len(arrs) == len(groups)

    return arrs, groups
