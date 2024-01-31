"""
Let chatGPT to determin a given sentence belongs to which 
class(es) of sentiment
"""

from openai import OpenAI
import os
import time
import json
from typing import Any
import pandas as pd
import torch

os.environ["OPENAI_API_KEY"] = "sk-ekCta1oLla8suXYzD53dCc393e6642CbAd2b181b1dF89416"
BASEURL = "https://oneapi.xty.app/v1"

# labels in original, ekman, group
CLASS_MAP = {
    "original": ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"],
    "ekman": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "group": ["ambiguous", "negative", "neutral", "positive"],
}

default_prompt = """Do sentiment classification with the given sentence.
        findout which class(es) the sentence belongs to.
        The possible classes are:{}.
        Give a list containing the index of the class(es) it belongs to, do not explain.\n
        For example:\n
        Input: ["I read on a different post that he died shortly after of internal injuries."]\n
        Output: [1,2]\n
        Input: ["Thank you friend"]\n
        Output: [3]\n
        Input: ["I'm not sure I have heard of this. Really interesting."]\n
        Output: [0]\n
        Input: ["I totally thought the same thing! I was like, oh honey nooooo!"]\n
        Output: [2]\n
        Input:
        """ # the few-shot examples are based on group class, change it if necessary

FREQ_DELAY = 0 # seconds


def GPTclassify(data:list, prompt = None, silent = True, temperature=0,
                class_type = "group")->list[int]:
    """
    let chatGPT judge which labels the given sentence belongs to
    and return the answers in list:
    [[idx1], [idx2], ...]
    parameter `prompt` gets the preferred judgement prompt.
    """
    Answers = []
    if prompt is None:
        prompt = default_prompt.format(CLASS_MAP[class_type])
        
    client = OpenAI(base_url=BASEURL)

    t = int(time.time())
    # test
    step=2
    for i in range(0,len(data),step):
        if not silent:
            print("{} - ".format(i),end="")
        response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          timeout=30,
          temperature=temperature,
          messages=[
            {
                # "role": "system", "content": prompt,
                "role": "user", "content": prompt+str(data[i:i+step if i+step<len(data) else len(data)])
            }
          ]
        )
        if not silent:
            print(response)
        # TODO 检测报错返回值
        answer = response.choices[0].message.content # get the text content of the answer
        # 分析返回值并转化为列表，可以用正则表达式？
        start_idx, end_idx = answer.find("["), answer.find("]")
        answer = json.loads(answer[start_idx:end_idx+1])

        # 转化为onehot
        # onehot = [[-10]*len(CLASS_MAP[class_type]) for _ in range(len(answer))]
        onehot = [0]*len(CLASS_MAP[class_type])
        # for j in range(len(answer)):
        #     for idx in answer[j]:
        #         onehot[j][idx] = 0
        for label in answer:
            onehot[label] = 1
        Answers.extend(onehot) #[[0 for not,1 for is],...]

        # hold to avoid rate limit(3RPM)
        t1 = int(time.time())
        while (t1-t < FREQ_DELAY*i):
            time.sleep(1)
            t1 = int(time.time())
    return Answers

# 13332
# test = ["I’m really sorry about your situation :( Although I love the names Sapphira, Cirilla, and Scarlett!",
#     "It's wonderful because it's awful. At not with.",
#     "Kings fan here, good luck to you guys! Will be an interesting game to watch!",
#     "I didn't know that, thank you for teaching me something today!",
#     "They got bored from haunting earth for thousands of years and ultimately moved on to the afterlife."]
# x = GPTclassify(data = test)
# print(x)

class LLMTokenizer:
    """
    A nonfunctional tokenizer for LLM
    """
    def __init__(self):
        # the tokenizer is ensembled with the pipeline
        pass 

    def __call__(self, txt, return_tensors) -> dict:
        # return the txt in a dict
        return {'txt': txt}

class LLM:
    def __init__(self, temperature = 0):
        """
        An LLM object works like a model object
        """
        # initialize
        pass

    def __call__(self, **kwds: Any) -> Any:
        # predict with input kwds(str)
        # refer to 'Test_Eval_Model.py' `self.model(**inputs)`
        # pred=self.pipeline.predict([kwds['txt']])
        pred = GPTclassify(data = [kwds['txt']], class_type="group")
        # convert to tensor
        pred = torch.tensor([pred], dtype=torch.float32, requires_grad=False)
        return pred

    def eval(self):
        """
        to stay consistent with bert etc.\\
        NO FUNCTIONALITY
        """
        print("LLM model does not have eval mode.")
        return