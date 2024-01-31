import os
import tqdm
import torch
import numpy as np
import pandas as pd
from typing import Any
from pypmml import Model
from sklearn2pmml import sklearn2pmml
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class BayesTokenizer:
    """
    A nonfunctional tokenizer for Bayes
    """
    def __init__(self):
        # the tokenizer is ensembled with the pipeline
        pass 

    def __call__(self, txt, return_tensors) -> dict:
        # return the txt in a dict
        return {'txt': txt}


class Bayes_Classifier():
    """
        An Bayes object works like a model object
    """
    def __init__(self, kernel_type = "mle", n_neighbors = 4, alpha = 1.1):
        if "mle" in kernel_type.lower():
            param = ("mle", OneVsRestClassifier(ComplementNB(alpha=alpha)))    # alpha hyperparam
            kernel_type = "mle"
        elif "knn" in kernel_type.lower():    
            param = ("knn", OneVsRestClassifier(KNeighborsClassifier(n_neighbors=n_neighbors)))    # n_neighbors hyperparam
            kernel_type = "knn"
        else:
            raise TypeError("no Bayes type named {}.".format(kernel_type))
        
        # load exited model
        mdpath = os.path.abspath('..')+"\\models\\Bayes-"+kernel_type+".pmml"
        if os.path.exists(mdpath):
            self.pipeline = Model.load(mdpath)
            self.train = False
            print("Load model Bayes-{} from {}".format(kernel_type, mdpath))
        else:
            self.train = True
            self.pipeline = Pipeline([('bow', CountVectorizer()), ('tfidf', TfidfTransformer()), param,])
            # self.pipeline = Pipeline([('bow', CountVectorizer()), ('tfidf', TfidfTransformer()), param])

        if self.train:
            # use group data to train Bayes model
            print("Training Bayes with group/train.tsv")
            train_data = pd.read_csv(os.getcwd()+"\\data\\group\\train.tsv", sep='\t', names=["txt","label"], usecols=[0,1])
            train_data["label"] = train_data["label"].apply(lambda x: x.split(','))
            mlb = MultiLabelBinarizer()
            train_data = train_data.join(pd.DataFrame(mlb.fit_transform(train_data.pop('label')), columns=mlb.classes_, index=train_data.index))
            for batch in tqdm.tqdm(range(0, len(train_data), 8000), desc="Training"):
                self.pipeline.fit(train_data['txt'][batch:batch+20], train_data.iloc[batch:batch+20,1:])
            # sklearn2pmml(self.pipeline, 'Bayes-'+kernel_type+'.pmml', with_repr=True, debug=False)
            # print("Saved model Bayes-{} to {}".format(kernel_type, mdpath))
            
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
        print("Bayes model does not have eval mode.")
        return
