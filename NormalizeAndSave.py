from xml.sax.xmlreader import IncrementalParser
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def normalize_and_save(df, drop_col_names, save_path, want_save=False):
    dropped_col_values = [df[col_name] for col_name in drop_col_names]
    new_df = df.drop(drop_col_names, inplace=False, axis=1)

    # normalize dataframe between 0 and 1
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df)

    # prepared dropped columns and rescaled columns to be able to
    # concatenate and create new dataframe
    dropped_df = pd.DataFrame.from_dict(
        dict(zip(drop_col_names, dropped_col_values)))
    scaled_df = pd.DataFrame.from_dict(dict(zip(df.columns, scaled_df.T)))

    # concatenate dropped and scaled
    concatenated_df = pd.concat([dropped_df, scaled_df], axis=1)

    # save as csv
    if want_save:
        concatenated_df.to_csv(save_path, index=False)

    return concatenated_df
