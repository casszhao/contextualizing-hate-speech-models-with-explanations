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


    # print(df.head(5))

    df.to_csv(new_csv_name, sep=',', index=False)
    return df



def addinfo(data_name, data_path):
    print(data_name)
    df = pd.read_csv(data_path, usecols=['text','label','prediction'] ) #, nrows=200
    # make pesudo prediction
    #df['Prediction'] = 1
    #df['text'] = df['text'].map(lambda x: x.lstrip('[CLS]',).rstrip('aAbBcC'))
    #df['text'] = df['text'].map(lambda x: x.lstrip('[SEP]',).rstrip('aAbBcC'))
    #df['text'] = df['text'].str.replace(r'[CLS]', '').str.replace(r'[SEP]', '')
    #bb_polarity =[]
    bb_subjective =[]
    for sent in df['text']:
        blob = TextBlob(sent)
        subjective = blob.sentiment.subjectivity
        # polarity = blob.sentiment.polarity
        #bb_polarity.append(polarity)
        bb_subjective.append(subjective)
    #print(len(df))
    #print(len(bb_polarity))
    #print(len(bb_subjective))
    #df['bb_polarity'] = bb_polarity
    df['Subjective Score'] = bb_subjective
    df['Data'] = data_name


    def results_tpye(row):
        if (row['label'] == 1 and row['prediction'] == 1):
            return 'True Positive'
        if row['label'] == 0 and row['prediction'] == 1:
            return 'False Positive'
        if row['label'] == 0 and row['prediction'] == 0:
            return 'True Negative'
        if row['label'] == 1 and row['prediction'] == 0:
            return 'False Negative'
        return 'Other'
    df['Result'] = df.apply(lambda row: results_tpye(row), axis=1)
    df = df.loc[(df['Result'] == 'True Positive') | (df['Result'] == 'False Positive')]

    print(df['Result'].value_counts())

    # only keep 'True Positive' and 'False Positive'
    return df

def combine_all_process(data_name, txt_path, csv_path):
    df = txt2csv(txt_path, csv_path)
    df_Ss = addinfo(data_name, csv_path)
    return df_Ss

D1 = combine_all_process('ws', './results/ws_bert.txt', './results/ws_bert.csv')
D2 = combine_all_process('wassem', './results/wassem_bert.txt', './results/wassem_bert.csv')
D3 = combine_all_process('AG10K', './results/AG10K_bert.txt', './results/AG10K_bert.csv')
D4 = combine_all_process('tweet50k', './results/tweet50k_bert.txt', './results/tweet50k_bert.csv')
D5 = combine_all_process('mt', './results/mt_bert.txt', './results/mt_bert.csv')
frames = [D1, D2, D3, D4, D5]
# D1_df = txt2csv('./results/ws_bert.txt', './results/ws_bert.csv')
# D1_df_Ss = addinfo('D1', './results/ws_bert.csv')

# D2_df = txt2csv('./results/ws_bert.txt', './results/ws_bert.csv')
# D2_df_Ss = addinfo('D2', './results/ws_bert.csv')

# D3_df = txt2csv('./results/ws_bert.txt', './results/ws_bert.csv')
# D3_df_Ss = addinfo('D3', './results/ws_bert.csv')


# label, text, Prediction, Subjective Score, Data, Result
df = pd.concat(frames)
print(df)
#df.to_csv('big_table_two_scores')

sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="Data", y="Subjective Score", hue="Result",
                 data=df, palette="Set3")
plt.show()
