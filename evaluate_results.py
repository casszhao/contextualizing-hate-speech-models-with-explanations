import pandas as pd
import numpy as np
import re
from sklearn.metrics import confusion_matrix, classification_report, f1_score

path = './results/eval_details_0_test_ws.txt'
new_csv_name = './results/ws_bert.csv'
def csv2txt(path, new_csv_name):

    with open(path, 'r') as f:
        content = f.readlines()


    label = []
    pre_non = []
    pre_toxic = []
    text = []

    for line in content:
        l, pre, t = line.strip().split('\t')
        number = re.findall(r"\d+\.?\d*",pre)
        label.append(int(l))
        pre_non.append(float(number[0]))
        pre_toxic.append(float(number[1]))
        text.append(t)


    df = pd.DataFrame({
        "label": label,
        "pre_non": pre_non,
        "pre_toxic": pre_toxic,
        "text": text
    })


    # print(df.head(5))

    df.to_csv(new_csv_name, sep=',', index=False)
    return df

ws_bert = csv2txt(path, new_csv_name)

conditions = [ ws_bert['pre_non'] > ws_bert['pre_toxic'], ws_bert['pre_non'] < ws_bert['pre_toxic'] ]
choices = [0, 1]
ws_bert['prediction'] = np.select(conditions, choices, default=np.nan)

confusion_matrix = confusion_matrix(ws_bert['label'], ws_bert['prediction'])
classification_report = classification_report(ws_bert['label'], ws_bert['prediction'])

TN = confusion_matrix[0][0]
FN = confusion_matrix[1][0]
TP = confusion_matrix[1][1]
FP = confusion_matrix[0][1]

print(TN)
print(FN)
print(TP)
print(FP)


print(classification_report)
print(f1_score(ws_bert['label'], ws_bert['prediction']))