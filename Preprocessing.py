from NormalizeAndSave import normalize_and_save
import pandas as pd


# load dataset
df = pd.read_csv('./dataset/bodyPerformanceI.csv')

# normalize dataset to be able to reach minimum while training
normalized_df = normalize_and_save(
    df, ['gender', 'class'], './dataset/bodyPerformanceNormalized.csv', True)
