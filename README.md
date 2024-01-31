# CS182 Project
# Performance evaluation of Reddit Comments using ML and NLP methods in Sentiment Analysis
> This is the final project for CS182, Introduction to Machine Learning in ShanghaiTech in 2023 fall.

## Code 
  - Before running anything, you should download two things: dataset and pytorch_model.bin. 
  - Dataset can be downloaded [here](https://github.com/Breadteeth/IML-Dataset/) and you should place them in the '/code/data' folder.
  - pytorch_model.bin can be downloaded [here](https://huggingface.co/SamLowe/roberta-base-go_emotions/blob/main/pytorch_model.bin). Notice that you should put the .bin file inside the '/models/roberta' folder (where there are already five files).
  - You can run by using the command line: `python main.py <model_name> <test_type>`, where test_type can be "acc" or "metrics" (with multiple metrics available). 
  - For example, the following commands are all legal: 
   1. `python main.py SVMlinear-original metrics`
   2. `python main.py SVMpoly-plutchik acc ` 
   3. `python main.py LLM-ekman acc`
   4. `python main.py Bayes-mle-original metrics`
  - Run in the '/code' working directory.
  - You can also use `python run_goemotions.py --taxonomy model_type `to evaluate BERT (original, Ekman, Plutchik).
  - RoBERTa model is available for the original task specifically. `python main.py roberta metrics` is allowed.


## Topic
使用Google [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) 数据集进行情感分类任务的实现与不同算法的表现比较.

## Contribution-Group 30
- 组员1 张晓夏：论文撰写与分析
- 组员2 齐修远：SVM、RoBERTa、LLM模型实现
- 组员3 滕孜信：Bayes+KNN/MLE模型实现
- 论文研究、模型考量、Presentation与其他工作：Equal Contribution

