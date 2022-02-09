from NormalizeAndSave import normalize_and_save
import pandas as pd


df = pd.read_csv('./dataset/bodyPerformanceI.csv')

normalize_and_save(df, ['gender', 'class'], './dataset/bodyPerformanceNormalized.csv')
