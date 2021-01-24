import pandas as pd
import re

def csv2txt():

    with open('./eval_details_400_dev_ws.txt', 'r') as f:
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

    df.to_csv('./eval_details_400_dev_ws.csv', sep=',', index=False)