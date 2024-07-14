import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm


splits = {'train': 'train.csv', 'validation': 'valid.csv', 'test': 'test.csv'}
df_train = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["train"])
df_valid = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["validation"])
df_test = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["test"])
df_train.rename(columns={"labels": "language"}, inplace=True)
df_valid.rename(columns={"labels": "language"}, inplace=True)
df_test.rename(columns={"labels": "language"}, inplace=True)
df = pd.concat([df_train, df_valid, df_test])

languages = df["language"].unique()
print("Number of rows: ", len(pd.concat([df_train, df_valid, df_test])))
print("Languages: ", languages)

encoder = LabelEncoder()
vectorizer = CountVectorizer()

train_X = df_train["text"]
train_X = vectorizer.fit_transform(train_X)
train_y = df_train["language"]
train_y = encoder.fit_transform(train_y)

batch_size = 1000
model = MultinomialNB()
for i in tqdm(range(0, train_X.shape[0], batch_size)):
    model.partial_fit(train_X[i:i+batch_size], train_y[i:i+batch_size], classes=np.unique(train_y))


tries = 100
correct = 0
test_X = df_test["text"]
test_X = vectorizer.transform(test_X)
test_y = df_test["language"]
test_y = encoder.transform(test_y)
for i in range(tries):
    pred = model.predict(test_X[i])
    if pred == test_y[i]:
        correct += 1

    if i % 10 == 0:
        print("Sentence: ", df_test["text"][i])
        print("Predicted: ", encoder.inverse_transform(pred)[0], "Actual: ", encoder.inverse_transform([test_y[i]])[0])
        print()

print("Accuracy: ", correct / tries)
