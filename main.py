import re
import pandas as pd
import sklearn


splits = {'train': 'train.csv', 'validation': 'valid.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["train"])
print(df.head())

print("Number of rows: ", len(df))
print("Languages: ", df["labels"].unique())