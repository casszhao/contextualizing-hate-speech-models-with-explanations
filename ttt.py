import pandas as pd
df = pd.read_csv('./data/white_supremacy/dev.tsv', sep='\t')


df['label'] = (df['label'] == 'hate').astype(int)

selected_columns = ['comment','label']

df.to_csv('./data/white_supremacy/dev.tsv', index=False, columns=selected_columns)