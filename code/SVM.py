from typing import Any
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from pypmml import Model
from sklearn2pmml import sklearn2pmml
import os
import tqdm
import torch

class SVMTokenizer:
    """
    A nonfunctional tokenizer for SVM
    """
    def __init__(self):
        # the tokenizer is ensembled with the pipeline
        pass 

    def __call__(self, txt, return_tensors) -> dict:
        # return the txt in a dict
        return {'txt': txt}

class SVM:
    def __init__(self, kernel_type = 'linear', deg = 2, gamma=0.3):
        """
        An SVM object works like a model object
        """
        # TODO initialize with necessary information

        if "linear" in kernel_type.lower():
            para = ('c', OneVsRestClassifier(
                        LinearSVC(multi_class="ovr",dual='auto')))
            kernel_type = "linear"
        elif "poly" in kernel_type.lower():
            para = ("svm_poly", OneVsRestClassifier(
                            SVC(kernel="poly", degree=deg, coef0=2, C=10)))
            kernel_type = "poly"
        elif "rbf" in kernel_type.lower():
            para = ("svm_clf", OneVsRestClassifier(
                            SVC(kernel="rbf", gamma=gamma, C=10)))
            kernel_type = "rbf"
        else:
            raise TypeError("no SVM type named {}.".format(kernel_type))
        
        # load model if it exists
        mdpath = os.path.abspath('..')+"\\models\\SVM-"+kernel_type+".pmml"
        if os.path.exists(mdpath):
            self.pipeline = Model.load(mdpath)
            self.train = False
            print("Loaded model SVM-{} from {}".format(kernel_type, mdpath))
        else:
            self.train = True
            self.pipeline = Pipeline([
                ('bow', CountVectorizer()),  
                ('tfidf', TfidfTransformer()),  
                para,
            ])
            print("Created model SVM-{}.".format(kernel_type))

        if self.train:
            # train model
            # use group data to train SVM
            print("Training SVM with group/train.tsv")
            train_data = pd.read_csv(os.getcwd()+"\\data\\group\\train.tsv", 
                                     sep='\t', names = ["txt", "label"],
                                     usecols=[0,1])
            train_data["label"] = train_data["label"].apply(lambda x: x.split(','))
            # use MultiLabelBinarizer to transform label
            mlb = MultiLabelBinarizer()
            train_data = train_data.join(pd.DataFrame(mlb.fit_transform(train_data.pop('label')),
                                  columns=mlb.classes_,
                                  index=train_data.index))
            # train
            for batch in tqdm.tqdm(range(0, len(train_data), 8000), desc="Training"):
                self.pipeline.fit(train_data['txt'][batch:batch+20], train_data.iloc[batch:batch+20,1:])
            # TODO save the model
            # sklearn2pmml(self.pipeline, 'SVM-'+kernel_type+'.pmml', with_repr=True, debug=False)
            # print("Saved model SVM-{} to {}".format(kernel_type, mdpath)) 

    def __call__(self, **kwds: Any) -> Any:
        # predict with input kwds(str)
        # refer to 'Test_Eval_Model.py' `self.model(**inputs)`
        # TODO
        pred=self.pipeline.predict([kwds['txt']])
        # convert to tensor
        pred = pd.array([(pred-1)*10])
        pred = torch.tensor(pred)
        return pred

    def eval(self):
        """
        to stay consistent with bert etc.\\
        NO FUNCTIONALITY
        """
        print("SVM model does not have eval mode.")
        return