import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
import re
nltk.download('averaged_perceptron_tagger')


def txt2csv(path, new_csv_name):

    with open(path, 'r') as f:
        content = f.readlines()

    label = []
    prediction = []
    pre_non = []
    pre_toxic = []
    text = []

    for line in content:
        l, pl, pre, t = line.strip().split('\t')
        number = re.findall(r"\d+\.?\d*",pre)
        label.append(int(l))
        prediction.append(int(pl))
        pre_non.append(float(number[0]))
        pre_toxic.append(float(number[1]))

        clean_txt = ''
        word_list = t.split(' ')
        for word in word_list[1:-1]:
            clean_txt = clean_txt + word + ' '

        text.append(clean_txt)

    df = pd.DataFrame({
        "label": label,
        "prediction": prediction,
        "pre_non": pre_non,
        "pre_toxic": pre_toxic,
        "text": text
    })


    df.to_csv(new_csv_name, sep=',', index=False)
    #print(df)
    return df



def addinfo(data_name, data_path):
    print(data_name)
    df = pd.read_csv(data_path, usecols=['text','is_hate'], sep='\t') #, nrows=200
    # if not contain, use df[df['text']
    df = df[df['text'].str.contains(
        "muslim|jew|jews|white|islam|blacks|muslims|women|whites|gay|black|democat|islamic|allah|jewish|lesbian|transgender|race|brown|woman|mexican|religion|homosexual|homosexuality|africans")]
    print(len(df))
    bb_subjective =[]
    for sent in df['text']:
        blob = TextBlob(sent)
        subjective = blob.sentiment.subjectivity
        # polarity = blob.sentiment.polarity
        #bb_polarity.append(polarity)
        bb_subjective.append(subjective)
    df['Subjective Score'] = bb_subjective
    df['Data'] = data_name

    return df



D1 = addinfo('WS', './data/white_supremacy/train.tsv')
#D2 = combine_all_process('AG10K', './results/AG10K_bert.txt', './results/AG10K_bert.csv')
D3 = addinfo('Twitter 15k', './data/wassem/train.tsv')
D3.to_csv('d3.csv')
D4 = addinfo('Twitter 50k', './data/tweet50k/train.tsv')
D4.to_csv('d4.csv')
D5 = addinfo('Wiki', './data/multi-label/train.tsv')
frames = [D1, D3, D4, D5]

# label, text, Prediction, Subjective Score, Data, Result
df = pd.concat(frames)

sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="Data", y="Subjective Score", hue="is_hate",
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
