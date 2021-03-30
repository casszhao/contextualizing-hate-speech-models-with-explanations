import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/zhixue/Downloads/DATA/80k_tweets/merged_retrieved.csv')
print(len(df))

df = df[df.label != 'spam']
print(len(df))

data = [df["text"], df["label"]]
headers = ["text", "is_hate"]
df = pd.concat(data, axis=1, keys=headers)
df.to_csv('./data/tweet42k/tweet42k.csv')

# change labels
df = df.replace({'is_hate': {'normal': int(0), 'abusive': int(1), 'hateful': int(1)}})
print(df.head())

df = df.dropna()
print(len(df))

print(df['is_hate'].value_counts())

train, tes = train_test_split(df, train_size=0.8, stratify=df['is_hate'])
dev, test = train_test_split(tes, test_size=0.5, stratify=tes['is_hate'])

print('train: ', len(train))
print(train['is_hate'].value_counts())
print('test: ', len(test))
print(test['is_hate'].value_counts())
print('dev: ', len(dev))
print(dev['is_hate'].value_counts())


train.to_csv('./data/tweet42k/train.tsv')
test.to_csv('./data/tweet42k/test.tsv')
dev.to_csv('./data/tweet42k/dev.tsv')
