import pandas as pd

df = pd.read_csv('./Gab/Gab_test.csv', usecols=['id', 'comment', 'label'])

print(df['label'].value_counts())

df.columns = ['doc_id','text','is_hate']
df.to_csv('./Gab/test.tsv', sep='\t', index=False)

print(df.head(3))