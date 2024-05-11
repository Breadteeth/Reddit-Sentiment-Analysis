# Performance evaluation of Reddit Comments using ML and NLP methods in Sentiment Analysis
> This is originally from our final project for CS182, Introduction to Machine Learning in ShanghaiTech in 2023 fall. After the course, we decided to continue to deepen and expand our research.

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
Implementation of emotion classification task using Google [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) data set and performance comparison of different algorithms.

## Contribution-Group 30
- Team member 1 Xiaoxia Zhang: paper writing and analysis
- Team Member 2 Xiuyuan Qi: Implementation of SVM, RoBERTa, and LLM models
- Team member 3 Zixin Teng: Bayes+KNN/MLE model implementation
- Thesis research, model consideration, Presentation and other work: Equal Contribution

