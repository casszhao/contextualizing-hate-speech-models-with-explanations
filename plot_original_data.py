import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
import re
nltk.download('averaged_perceptron_tagger')


def addinfo(data_name, data_path):
    print(data_name)
    df = pd.read_csv(data_path, usecols=['text','is_hate'], sep='\t') #, nrows=200　＃, lineterminator='\n'
    # if not contain, use df[~df['text']

    df = df[~df['text'].str.lower().str.contains(
    #df = df[~df['text'].str.contains(
        "muslim|jew|jews|white|islam|blacks|muslims|women|whites|gay|black|democat|islamic|allah|jewish|lesbian|transgender|race|brown|woman|mexican|religion|homosexual|homosexuality|africans")]
    df['is_hate'] = df['is_hate'].replace([0, 1], ['Not Toxic', 'Toxic'])
    print(len(df))

    bb_subjective = []
    #sent128_list = []

    for sent in df['text']:
        # print(sent)
        # sent = sent.split(' ', 60)[:60]
        # sent = ' '.join(sent)
        sent = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", sent).split())

        remove_RT = lambda x: re.compile('RT @').sub('@', x, count=1)
        sent = remove_RT(sent)
        # sent128_list.append(sent128)

        #sent128 = sent128.lower() #.replace('[^\w\s]','')
        # print(sent128)
        # break

        blob = TextBlob(sent)
        subjective = blob.sentiment.subjectivity
        # polarity = blob.sentiment.polarity
        #bb_polarity.append(polarity)
        bb_subjective.append(subjective)
    df['Subjectivity Score'] = bb_subjective
    df['Data'] = data_name

    return df #, sent128_list


# df, sent128 = addinfo('Twitter 18k', './data/wassem/train.tsv')
#
# print(sent128)
# stop


D1 = addinfo('WS', './data/white_supremacy/train.tsv')
D3 = addinfo('Twitter 18k', './data/wassem/train.tsv')
D4 = addinfo('Twitter 42k', 'data/tweet42k/no_pro/train.tsv')
D5 = addinfo('Wiki', './data/multi-label/train.tsv')

#D2 = combine_all_process('AG10K', './results/AG10K_bert.txt', './results/AG10K_bert.csv')

# D3.to_csv('d3.csv')

# D4.to_csv('d4.csv')

frames = [D1, D3, D4, D5]

# label, text, Prediction, Subjective Score, Data, Result
df = pd.concat(frames)

df.to_csv('all_idf_comment.csv')

sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="Data", y="Subjectivity Score", hue="is_hate",
                 data=df, palette="Set3")
# ax.despine(left=True)
# plt.legend(loc='upper left')
#plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# ax.set_xlabel('xlabel')
# ax.set_ylabel('ylabel')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position

# Put a legend to the right side
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1), ncol=1)#bbox_to_anchor=(1.4, 1),左右 越大越右 上下 越大约上
plt.show()
