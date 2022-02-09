from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def normalize_and_save(df, drop_col_names, save_path):
    dropped_col_values = []

    # drop cols inside drop_col_names list ['gender', 'class']
    # because we cannot normalize dataset with str values
    for col_name in drop_col_names:
        try:
            dropped_col_values.append(df[col_name])
            df.drop(col_name, inplace=True, axis=1)
        except:
            print('Column not found:  ', col_name)
    
    # normalize dataframe between 0 and 1
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df)

    # prepared dropped columns and rescaled columns to be able to
    # concatenate and create new dataframe
    dropped_df = pd.DataFrame.from_dict(dict(zip(drop_col_names, dropped_col_values)))
    scaled_df = pd.DataFrame.from_dict(dict(zip(df.columns, scaled_df.T)))

    # concatenate dropped and scaled 
    concatenated_df = pd.concat([dropped_df, scaled_df], axis=1)

    # save as csv 
    concatenated_df.to_csv(save_path)

    return concatenated_df
