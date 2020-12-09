import pandas as pd

df = pd.read_csv('./Gab/Gab_validation.csv', usecols=['id', 'comment', 'label'])

df.columns = ['doc_id','text','is_hate']
df.to_csv('./Gab/dev.tsv', sep='\t', index=False)

print(df.head(3))