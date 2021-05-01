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
        number = re.findall(r"\d+\.?\d*", pre)
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

    df = df[~df['text'].str.contains("muslim|jew|jews|white|islam|blacks|muslims|women|whites|gay|black|democat|islamic|allah|jewish|lesbian|transgender|race|brown|woman|mexican|religion|homosexual|homosexuality|africans")]

    df.to_csv(new_csv_name, sep=',', index=False)
    # print(df)
    return df


def addinfo(data_name, data_path):
    print(data_name)
    df = pd.read_csv(data_path, usecols=['text', 'label', 'prediction']).dropna()  # , nrows=200
    # df = df[df['text'].str.lower().str.contains(
    #     # df = df[~df['text'].str.contains(
    #     "muslim|jew|jews|white|islam|blacks|muslims|women|whites|gay|black|democat|islamic|allah|jewish|lesbian|transgender|race|brown|woman|mexican|religion|homosexual|homosexuality|africans")]

    bb_subjective = []
    for sent in df['text']:
        sent = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", sent).split())
        remove_RT = lambda x: re.compile('\#').sub('', re.compile('RT @').sub('@', x, count=1).strip())
        sent = remove_RT(sent)

        blob = TextBlob(sent)
        subjective = blob.sentiment.subjectivity
        # polarity = blob.sentiment.polarity
        # bb_polarity.append(polarity)
        bb_subjective.append(subjective)
    df['Subjectivity Score'] = bb_subjective
    df['Data'] = data_name

    def results_tpye(row):
        if (row['label'] == 1 and row['prediction'] == 1):
            return 'TPwoIT'
        if row['label'] == 0 and row['prediction'] == 1:
            return 'FPwoIT'
        if row['label'] == 0 and row['prediction'] == 0:
            return 'True Negative'
        if row['label'] == 1 and row['prediction'] == 0:
            return 'FNwIT'
        return 'Other'

    df['Result'] = df.apply(lambda row: results_tpye(row), axis=1)
    df = df.loc[(df['Result'] == 'TPwoIT') | (df['Result'] == 'FPwoIT')]
    print(df['Result'].value_counts())
    # only keep 'True Positive' and 'False Positive'
    return df


def combine_all_process(data_name, txt_path, csv_path):
    df = txt2csv(txt_path, csv_path)
    df_Ss = addinfo(data_name, csv_path)
    return df_Ss


D1 = combine_all_process('WS', './results/ws_bert_9.txt', './results/ws_bert_9_plot.csv')
# D2 = combine_all_process('AG10K', './results/AG10K_bert.txt', './results/AG10K_bert.csv')
D3 = combine_all_process('Twitter 18k', './results/wassem_bert_8.txt', './results/wassem_bert_4_plot.csv')
D4 = combine_all_process('Twitter 42k', './results/tweet42k_bert_7.txt', './results/tweet50k_bert_6_plot.csv')
D5 = combine_all_process('Wiki', './results/mt_bert_0.txt', './results/mt_bert_0_plot.csv')
frames = [D1, D3, D4, D5]

# label, text, Prediction, Subjective Score, Data, Result
df = pd.concat(frames)

df.to_csv('./results/TP_FP.csv')

sns.set_theme(style="whitegrid")

my_palette = {"FPwoIT": "g", "TPwoIT": "y"}
# sns.boxplot(x=df["species"], y=df["sepal_length"], palette=my_pal)

ax = sns.boxplot(x="Data", y="Subjectivity Score", hue="Result",
                 data=df, palette="Set3", hue_order=["FPwoIT", "TPwoIT"])  # "Set3"
# ax.despine(left=True)
# plt.legend(loc='upper left')
# plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# ax.set_xlabel('xlabel')
# ax.set_ylabel('ylabel')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position

# Put a legend to the right side
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), ncol=1)  # bbox_to_anchor=(1.4, 1),左右 越大越右 上下 越大约上
plt.show()
