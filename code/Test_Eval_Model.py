"""
need a Class for testing on different models
and another for result analysis
"""


from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import csv
import torch
from tqdm import tqdm
import numpy as np
##files from git
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from utils import compute_metrics
from LLM import LLMTokenizer, LLM
import Bayes
import SVM

class Eval:
    def __init__(self, model_name):
        """
        bert and roberta are as usual,\\
        pass object like SVM through `model_object`\\
        """
        self.model_name = model_name

        if "bert" in model_name.split("-"):
            self.model_path=os.path.abspath('..')+ "\\models\\" + model_name
            self.tokenizer = BertTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True)
            self.model = BertForMultiLabelClassification.from_pretrained(
                self.model_path,
                local_files_only=True)
        elif "roberta" in model_name.split("-"):
            self.model_path=os.path.abspath('..')+ "\\models\\" + model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True)
        elif "Bayes" in model_name:
            self.model_path = ""
            self.tokenizer = Bayes.BayesTokenizer()
            self.model = Bayes.Bayes_Classifier(kernel_type=model_name)
        elif "SVM" in model_name:
            self.model_path=""
            self.tokenizer = SVM.SVMTokenizer()
            # create a pipeline
            self.model = SVM.SVM(kernel_type=model_name)
        elif "LLM" in model_name:
            self.model_path=""
            self.tokenizer = LLMTokenizer()
            # create a pipeline
            self.model = LLM(temperature=0)
        else:
            raise NotImplementedError("Test for {} not implemented yet.".format(model_name))
        self.model.eval()
        # TODO use cuda?
        # print necessary information
        print("Loaded model: {} from {}.".format(self.model_name, self.model_path))

    def __load_data(self, data_path):
        data = [] #[("sentence", {label1, label2,...})]
        with open(data_path, encoding="utf8") as f:
            tsvreader = csv.reader(f, delimiter='\t')
            # store the examples and labels in data
            for line in tsvreader:
                #separate labels in label[1] separated by ','
                labels = [int(x) for x in line[1].split(',')]
                data.append((line[0], set(labels)))
        # check if data and model matches 
        md_type = self.model_name.split('-')[-1] # original, ekman, group
        data_type = data_path.split('\\')[-2] # original, ekman, group
        if md_type != data_type:
            print("\033[33;40mWarning: model type and data type do not match!\033[0m")
        return data

    def predict(self, data, threshold=0.3):
        results = []
        for pair in tqdm(data, desc="Predicting"):
            txt = pair[0]
            inputs = self.tokenizer(txt,return_tensors="pt")
            # print(inputs) #
            outputs = self.model(**inputs)
            if "LLM" in self.model_name:
                scores = outputs
            else: 
                scores = torch.sigmoid(outputs[0])
            # scores =  1 / (1 + torch.exp(-outputs[0]))  # Sigmoid
            # threshold = threshold #TODO 之后可以对不同的threshold进行测试
            for item in scores:
                labels = []
                scores = []
                for idx, s in enumerate(item):
                    if s > threshold:
                        labels.append(idx)
                        scores.append(s.data) # 这里和原文比改为了s.data，去掉了pt
                results.append({"labels": labels, "scores": scores})
        return results

    def get_acc(self, data_path):
        
        data = self.__load_data(data_path)
        results = self.predict(data)
        true_count = 0
        for i in range(len(results)):
            predict=set(results[i]['labels'])
            ground = set(data[i][1])
            if ground == predict:
                true_count+=1
        acc = true_count/len(results)
        return acc
    
    def get_metrics(self, data_path):
        data = self.__load_data(data_path)
        data = data[:50] #FIXME for test
        type_num_map = {
            "original": 28,
            "ekman": 7,
            "group": 4
        }
        type_num = type_num_map[data_path.split('\\')[-2]]

        results = self.predict(data)
        preds = None
        for res in results:
            one_hot_label = [0] * type_num
            for l in res['labels']:
                one_hot_label[l] = 1
            if preds is None:
                preds = np.array([one_hot_label])
            else:
                preds = np.append(preds, [one_hot_label], axis=0)
        ground = None
        for d in data:
            one_hot_label = [0] * type_num
            for l in d[1]:
                one_hot_label[l] = 1
            if ground is None:
                ground = np.array([one_hot_label])
            else:
                ground = np.append(ground, [one_hot_label], axis=0)
        
        result = compute_metrics(ground, preds)
        return result
    
    def get_classed_metrics(self, data_path):
        data = self.__load_data(data_path) # [("text",{l1,...})]
        data = data[:500] #FIXME for test
        type_num_map = {
            "original": 28,
            "ekman": 7,
            "group": 4
        }
        type_num = type_num_map[data_path.split('\\')[-2]]

        results = self.predict(data) #[{"labels":[l1,...], "scores":[tensor,]}]
        # calculate confusion matrix
        tp = np.zeros([type_num])
        fp = np.zeros([type_num])
        tn = np.zeros([type_num])
        fn = np.zeros([type_num])
        preds = []
        for res in results:
            one_hot_label = [0] * type_num
            for l in res['labels']:
                one_hot_label[l] = 1
            preds.append([one_hot_label])
        ground = []
        for d in data:
            one_hot_label = [0] * type_num
            for l in d[1]:
                one_hot_label[l] = 1
            ground.append([one_hot_label])
        preds = np.array(preds)
        ground = np.array(ground)
        for i in range(preds.shape[0]):
            tp+=np.bitwise_and(ground[i],preds[i]).reshape(-1)
            tn+=np.bitwise_and(~ground[i],~preds[i]).reshape(-1)
            fp+=np.bitwise_and(~ground[i],preds[i]).reshape(-1)
            fn+=np.bitwise_and(ground[i],~preds[i]).reshape(-1)
        result = {}
        result["precision"] = tp/(tp+fp)
        result["recall"] = tp/(tp+fn)
        result["f1"] = 2*result["precision"]*result["recall"]/(result["precision"]+result["recall"])
        return result

