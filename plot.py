import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
nltk.download('averaged_perceptron_tagger')


data_path = './results/ws_bert.csv'
def addinfo(data_name, data_path):
    print(data_name)
    df = pd.read_csv(data_path, usecols=['text','label'] ) #, nrows=200
    # make pesudo prediction
    df['Prediction'] = 1
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
        if (row['label'] == 1 and row['Prediction'] == 1):
            return 'True Positive'
        if row['label'] == 0 and row['Prediction'] == 1:
            return 'False Positive'
        if row['label'] == 0 and row['Prediction'] == 0:
            return 'True Negative'
        if row['label'] == 1 and row['Prediction'] == 0:
            return 'False Negative'
        return 'Other'
    df['Result'] = df.apply(lambda row: results_tpye(row), axis=1)
    df = df.loc[(df['Result'] == 'True Positive') | (df['Result'] == 'False Positive')]

    print(df['Result'].value_counts())

    # only keep 'True Positive' and 'False Positive'
    return df



D1 = addinfo('D1', data_path)
D2 = addinfo('D2', data_path)
D3 = addinfo('D3', data_path)

frames = [D1, D2, D3]
# label, text, Prediction, Subjective Score, Data, Result
df = pd.concat(frames)
print(df)
#df.to_csv('big_table_two_scores')

sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="Data", y="Subjective Score", hue="Result",
                 data=df, palette="Set3")
plt.show()
