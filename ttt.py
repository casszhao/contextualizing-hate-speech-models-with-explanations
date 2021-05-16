import torch
#from bert.tokenization import BertTokenizer
import pandas as pd

from transformers import RobertaForSequenceClassification, RobertaModel, RobertaConfig, RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
a = tokenizer.convert_ids_to_tokens([11,11,22])
print(a)




class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

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

