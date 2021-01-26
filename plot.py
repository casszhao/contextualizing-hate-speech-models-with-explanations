import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv('results/ws_bert.csv')
bb_polarity =[]
bb_subjective =[]

for sent in df['Text']:
    blob = TextBlob(sent)
    polarity = blob.sentiment.polarity
    subjective = blob.sentiment.subjectivity
    bb_polarity.append(polarity)
    bb_subjective.append(subjective)

print(len(df))
print(len(bb_polarity))
print(len(bb_subjective))
df['bb_polarity'] = bb_polarity
df['bb_subjective'] = bb_subjective

df.to_csv('big_table_two_scores')

sns.set_theme(style="whitegrid")

ax = sns.boxplot(x="Dataset", y="bb_polarity", hue="Data",
                 data=df, palette="Set3")

plt.show()

ax = sns.boxplot(x="Dataset", y="bb_subjective", hue="Data",
                 data=df, palette="Set3")

plt.show()