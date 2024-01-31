import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# read the data from ('.\\roberta.json')
roberta = pd.read_json('.\\roberta.json').T
svm = pd.read_json('.\\SVM.json').T
# read bert data from txt
bert_ekman, bert_group, bert_original = None, None, None
for i in ["original", "ekman", "group"] :
    bert = pd.read_csv('.\\bert-base-cased-goemotions-'+i+'\\eval_results.txt', sep=" = ", 
                            names=['bert-'+i],index_col=0,engine='python')
    # delete the post fix(_i) of row name
    bert.rename(index=lambda x: x.replace("_{}".format(i),""), inplace=True)
    bert = bert.T
    globals()["bert_"+i] = bert

datalist = [roberta, svm, bert_ekman, bert_group, bert_original]
data = pd.concat(datalist, join = "inner")
data.rename(columns={"index":"model"}, inplace=True)
print(data)

## accuarcy
plt.figure(figsize=(10,6))
plt.title("Accuracy")
plt.xlabel("model")
plt.ylabel("accuracy")
plt.ylim(0, 1)
plt.bar(data.index, data["accuracy"])
plt.savefig("accuracy.png")
plt.show()
