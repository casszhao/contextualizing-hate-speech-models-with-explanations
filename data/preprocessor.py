import pandas as pd

df = pd.read_csv('binary_multi-label_train.csv', usecols=['Unnamed: 0', 'comment', 'binary'])

df.columns = ['doc_id','text','is_hate']
df.to_csv('./multi-label/train.tsv', sep='\t', index=False)

print(df.head(3))