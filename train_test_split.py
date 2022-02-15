from sklearn.model_selection import train_test_split
import pandas as pd


# load dataset
data = pd.read_csv('./dataset/bodyPerformanceNormalized.csv')

# take genders as labels
genders = data['gender'] # labels of the whole dataset

# drop genders from train dataset
data.drop('gender', inplace=True, axis=1) # drop labels from dataset to make it as list of features

# split into train and test sets (90%, 10% relatively)
X_train, X_test, y_train, y_test = train_test_split(data, genders, test_size=0.1)

# concatenate and save test examples for future usage
concatenated_test_df = pd.concat([X_test, y_test], axis=1) # concatenate labels and features
concatenated_test_df.to_csv('./dataset/test.csv', index=False) # save as csv files

# concatenate and save train examples for future usage
concatenated_train_df = pd.concat([X_train, y_train], axis=1) # concatenate labels and features
concatenated_train_df.to_csv('./dataset/train.csv', index=False) # save as csv files
