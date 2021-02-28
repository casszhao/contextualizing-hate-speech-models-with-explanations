import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np

# csv = pd.read_csv('./GabHateCorpus_annotations.tsv', sep='\t')
#
# df_hd_cv = csv.groupby('ID')['HD', 'CV'].mean().round()
# # print('length of df_hd_cv', len(df_hd_cv))
# # print(df_hd_cv.tail())
# # print(df_hd_cv.iloc[-2])
#
# non_duplicate = csv.drop_duplicates(subset=['ID'])
# # print('after remove')
# # print('non_duplicate', len(non_duplicate))
# non_duplicate = non_duplicate.replace(np.nan, 0)
# # print(non_duplicate.iloc[-2])
# # print(non_duplicate.tail())
# for index, row in df_hd_cv.iterrows():
#
#     non_duplicate.iloc[index]['HD'] = row['HD']
#     non_duplicate.iloc[index]['CV'] = row['CV']
#
# # print('-------')
# # print(non_duplicate.iloc[-2])
#
# # non_duplicate.to_csv('./new_data.tsv', index=False, sep='\t')
#
# rows_with_nan = [index for index, row in non_duplicate.iterrows() if row.isnull().any()]
#
# print(rows_with_nan[:10])
#
# # print(non_duplicate.iloc(rows_with_nan[0]))
#
# print('non_duplicate', len(non_duplicate))
# print(non_duplicate)
#
# print(non_duplicate.isnull().values.any())
# print(non_duplicate.isnull().sum().sum())
# non_duplicate.to_csv('./new_data.tsv', index=False, sep='\t')
non_duplicate = pd.read_csv('./new_data.tsv', sep='\t')
# print(non_duplicate.isnull().sum().sum())
# rows_with_nan = [index for index, row in non_duplicate.iterrows() if row.isnull().any()]
# print(rows_with_nan[:10])
# print(non_duplicate.iloc[rows_with_nan[0]])


train, test = train_test_split(non_duplicate, test_size=0.2, random_state=42, stratify=non_duplicate['HD'])
validation, test = train_test_split(test, test_size=0.5, random_state=42, stratify=test['HD'])

# {"text_id":31287737,"Text":"How is that one post not illegal? He is calling for someone to commit a specific crime or he will do it himself. ",
# "im":0,"cv":0,"ex":0,"hd":0,"mph":0,"gen":0,"rel":0,"sxo":0,"rae":0,"nat":0,"pol":0,"vo":0,"idl":0}

print(train)

for index, row in train.iterrows():
    df_dict = {}
    df_dict['text_id'] = row['ID']
    df_dict['Text'] = row['Text']
    df_dict['im'] = row['IM']
    df_dict['cv'] = row['CV']
    df_dict['ex'] = row['EX']
    df_dict['hd'] = row['HD']
    df_dict['mph'] = row['MPH']
    df_dict['gen'] = row['GEN']
    df_dict['rel'] = row['REL']
    df_dict['sxo'] = row['SXO']
    df_dict['rae'] = row['RAE']
    df_dict['nat'] = row['NAT']
    df_dict['pol'] = row['POL']
    df_dict['vo'] = row['VO']
    df_dict['idl'] = row['IDL']




    with open('../majority_gab_dataset_25k/train.jsonl','a') as f:
        # f.write("%s\n" % str(df_dict))
        f.write("%s\n" % str(json.dumps(df_dict)))


for index, row in validation.iterrows():
    df_dict = {}
    df_dict['text_id'] = row['ID']
    df_dict['Text'] = row['Text']
    df_dict['im'] = row['IM']
    df_dict['cv'] = row['CV']
    df_dict['ex'] = row['EX']
    df_dict['hd'] = row['HD']
    df_dict['mph'] = row['MPH']
    df_dict['gen'] = row['GEN']
    df_dict['rel'] = row['REL']
    df_dict['sxo'] = row['SXO']
    df_dict['rae'] = row['RAE']
    df_dict['nat'] = row['NAT']
    df_dict['pol'] = row['POL']
    df_dict['vo'] = row['VO']
    df_dict['idl'] = row['IDL']

    with open('../majority_gab_dataset_25k/dev.jsonl','a') as f:
        f.write("%s\n" % str(json.dumps(df_dict)))

for index, row in test.iterrows():
    df_dict = {}
    df_dict['text_id'] = row['ID']
    df_dict['Text'] = row['Text']
    df_dict['im'] = row['IM']
    df_dict['cv'] = row['CV']
    df_dict['ex'] = row['EX']
    df_dict['hd'] = row['HD']
    df_dict['mph'] = row['MPH']
    df_dict['gen'] = row['GEN']
    df_dict['rel'] = row['REL']
    df_dict['sxo'] = row['SXO']
    df_dict['rae'] = row['RAE']
    df_dict['nat'] = row['NAT']
    df_dict['pol'] = row['POL']
    df_dict['vo'] = row['VO']
    df_dict['idl'] = row['IDL']

    with open('../majority_gab_dataset_25k/test.jsonl','a') as f:
        f.write("%s\n" % str(json.dumps(df_dict)))
