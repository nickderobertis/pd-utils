import re
from typing import List

import pandas as pd
from pandasql import PandaSQL


def select_rows_by_condition_on_columns(df: pd.DataFrame, cols: List[str],
                                        condition: str = "== 1", logic: str = "or"):
    """
    Selects rows of a pandas dataframe by evaluating a condition on a subset of the dataframe's columns.

    :param df:
    :param cols: column names, the subset of columns on which to evaluate conditions
    :param condition: needs to contain comparison operator and right hand side of comparison. For example,
       '== 1' checks for each row that the value of each column is equal to one.
    :param logic: 'or' or 'and'. With 'or', only one of the columns in cols need to match the condition
        for the row to be kept. With 'and', all of the columns in cols need to match the condition.
    :return:
    """
    # First eliminate spaces in columns, this method will not work with spaces
    new_cols = [col.replace(" ", "_").replace(".", "_") for col in cols]
    df.rename(
        columns={col: new_col for col, new_col in zip(cols, new_cols)}, inplace=True
    )

    # Now create a string to query the dataframe with
    logic_spaces = " " + logic + " "
    query_str = logic_spaces.join(
        [str(col) + condition for col in new_cols]
    )  #'col1 == 1, col2 == 1', etc.

    # Query dataframe
    outdf = df.query(query_str).copy()

    # Rename columns back to original
    outdf.rename(
        columns={new_col: col for col, new_col in zip(cols, new_cols)}, inplace=True
    )

    return outdf


def sql(df_list: List[pd.DataFrame], query: str):
    """
    Convenience function for running a pandasql query. Keeps track of which variables are of
    datetime type, and converts them back after running the sql query.

    :Notes:

    Ensure that dfs are passed in the order that they are used in the query.

    :param df_list:
    :param query:
    :return:
    """
    # TODO [#8]: add example in docs for sql

    # Pandasql looks up tables by names given in query. Here we are passed a list of dfs without names.
    # Therefore we need to extract the names of the tables from the query, then assign
    # those names to the dfs in df_list in the locals dictionary.
    table_names = _extract_table_names_from_sql(query)
    for i, name in enumerate(table_names):
        locals().update({name: df_list[i]})

    # Get date variable column names
    datevars: List[str] = []
    for d in df_list:
        datevars += _get_datetime_cols(d)
    datevars = list(set(datevars))  # remove duplicates

    merged = PandaSQL()(query)

    # Convert back to datetime
    for date in [d for d in datevars if d in merged.columns]:
        merged[date] = pd.to_datetime(merged[date])
    return merged


def _extract_table_names_from_sql(query):
    """ Extract table names from an SQL query. """
    # a good old fashioned regex. turns out this worked better than actually parsing the code
    tables_blocks = re.findall(r'(?:FROM|JOIN)\s+(\w+(?:\s*,\s*\w+)*)', query, re.IGNORECASE)
    tables = [tbl
              for block in tables_blocks
              for tbl in re.findall(r'\w+', block)]
    return list(dict.fromkeys(tables).keys())  # remove duplicates, keeping order


def _get_datetime_cols(df):
    """
    Returns a list of column names of df for which the dtype starts with datetime
    """
    dtypes = df.dtypes
    return dtypes.loc[dtypes.apply(lambda x: str(x).startswith('datetime'))].index.tolist()