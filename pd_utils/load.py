import os

import pandas as pd
from sas7bdat import SAS7BDAT


def load_sas(filepath: str, csv: bool = True, **read_csv_kwargs):
    """
    Loads sas sas7bdat file into a pandas DataFrame.

    :param filepath: str of location of sas7bdat file
    :param csv: when set to True, saves a csv version of the data in the same directory as the sas7bdat.
                Next time load_sas will load from the csv version rather than sas7bdat, which speeds up
                load times about 3x. If the sas7bdat file is modified more recently than the csv,
                the sas7bdat will automatically be loaded and saved to the csv again.
    :param read_csv_kwargs: kwargs to pass to pd.read_csv if csv option is True
    :return:
    """
    sas_name = os.path.basename(filepath)  # e.g. dsename.sas7bdat
    folder = os.path.dirname(filepath)  # location of sas file
    filename, extension = os.path.splitext(sas_name)  # returns ('dsenames','.sas7bdat')
    csv_name = filename + ".csv"
    csv_path = os.path.join(folder, csv_name)

    if os.path.exists(csv_path) and csv:
        if os.path.getmtime(csv_path) > os.path.getmtime(
            filepath
        ):  # if csv was modified more recently
            # Read from csv (don't touch sas7bdat because slower loading)
            try:
                return pd.read_csv(csv_path, encoding="utf-8", **read_csv_kwargs)
            except UnicodeDecodeError:
                return pd.read_csv(csv_path, encoding="cp1252", **read_csv_kwargs)

    # In the case that there is no csv already, or that the sas7bdat has been modified more recently
    # Pull from SAS file
    df = SAS7BDAT(filepath).to_data_frame()
    # Write to csv file
    if csv:
        file_path = os.path.join(folder, filename)
        df.to_csv(file_path, index=False)
    return df