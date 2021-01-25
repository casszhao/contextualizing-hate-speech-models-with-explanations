import pandas as pd
import re

path = './data/eval_details_400_dev_ws.txt'
new_csv_name = './data/eval_details_400_dev_ws.csv'
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

csv2txt(path, new_csv_name)