def to_copy_paste(df, index=False, column_names=True):
    """
    Takes a dataframe and prints all of its data in such a format that it can be copy-pasted to create
    a new dataframe from the pandas.DataFrame() constructor.

    Required inputs:
    df: pandas dataframe

    Optional inputs:
    index: bool, True to include index
    column_names: bool, False to exclude column names
    """
    print("pd.DataFrame(data = [")
    for tup in df.iterrows():
        data = tup[1].values
        print(str(tuple(data)) + ",")
    last_line = "]"
    if column_names:
        last_line += ", columns = {}".format(
            [i for i in df.columns]
        )  # list comp to remove Index() around cols
    if index:
        last_line += ",\n index = {}".format(
            [i for i in df.index]
        )  # list comp to remove Index() around index
    last_line += ")"  # end command
    print(last_line)