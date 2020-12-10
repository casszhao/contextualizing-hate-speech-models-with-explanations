import pandas as pd
df = pd.read_csv('./data/Gab/test.tsv', header=0, sep='\t')



#df.columns = ['doc_id','text','is_hate']

#df['is_hate'] = (df['is_hate'] == 'hate').astype(int)

#selected_columns = ['comment','label']

#df.to_csv('./data/white_supremacy/dev.tsv', sep='\t', index=False)

print(df['is_hate'].value_counts())
