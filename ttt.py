import torch
from bert.tokenization import BertTokenizer
import pandas as pd

data = pd.read_csv('data/multi-label/train.tsv', sep='\t')

print(data['is_hate'].value_counts())


# tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
#
#
# tensor = ([101, 2105, 1019, 1003, 1997, 12029, 2015, 4823, 2006, 2028,
#            2754, 1012, 102, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#            0, 0, 0, 0, 0, 0, 0, 0])
#
# sent = tokenizer.convert_ids_to_tokens(tensor)
# print(sent)
# sent = [x.lower() for x in sent]
# print(sent)
# words = set(sent)
#
# def read_igw(in_file):
#     f = open(in_file, 'r')
#     line = f.readline()
#     res=set()
#     while line:
#         line = f.readline().strip()
#         if line.startswith("#") or len(line)==0:
#             continue
#         res.add(line)
#     print(' identifiers')
#     print(res)
#     return res
#
#
# idgw_file = "./data/identity_group_words.txt"
# igw=read_igw(idgw_file)
#
#
# inter = words.intersection(igw)
#
# print(len(inter))

