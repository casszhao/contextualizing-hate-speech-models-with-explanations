import torch
import argparse
from utils.config import configs, combine_args
#from bert.tokenization import BertTokenizer
import pandas as pd

from transformers import RobertaForSequenceClassification, RobertaModel, RobertaConfig, RobertaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--d1111111",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
args = parser.parse_args()
print(args)

parser2 = argparse.ArgumentParser()
parser2.add_argument("--d2222222222",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

args2 = parser2.parse_args()
print(args2)
combine_args(args2, args)
#args = args2
print('')
print(args)




#
# df = pd.read_csv('./data/wassem/train.tsv', sep = '\t')
#
# # toxic = df['is_hate'].value_counts()
# print(df['text'].apply(len).mean())

# df = pd.read_csv('data/tweet42k/no_pro/dev.tsv', lineterminator='\n')
# df.to_csv('./data/tweet42k/dev.tsv', sep='\t')
#
# df = pd.read_csv('data/tweet42k/no_pro/dev.tsv')
# print(df.head())

# a = torch.zeros([32, 128])
#
# shape = a.shape
#
# print(shape)
# print(shape[0])
#
# b = torch.ones([a.shape[0], 1])
#
#
# attention_mask = torch.cat([a, b], dim=1)
#
# print(attention_mask)
# print(attention_mask.shape)

# tensor1 = torch.tensor([[1,1,1],
#                        [2,2,2]])
#
#
# Ss = torch.empty(5, 1)
# print(Ss.size())
# print(Ss)
# Ss = Ss.unsqueeze(1).unsqueeze(2)
# print(Ss.size())
# IDW = torch.empty([5, 1], dtype=torch.long)
# IDW[0, 0] = 1
# print('IDW', IDW.type())
# stop
# tensor2 = torch.tensor([[0.5],
#             [1]])
#
# tensor3 = tensor1 * tensor2
# print(tensor3)


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

