import os


texts = [
    "Hey that's a thought! Maybe we need [NAME] to be the celebrity vaccine endorsement!",
    "it's happened before?! love my hometown of beautiful new ken 😂😂",
    "I love you, brother.",
    "Troll, bro. They know they're saying stupid shit. The motherfucker does nothing but stink up libertarian subs talking shit",
]

import Test_Eval_Model

md = Test_Eval_Model.Eval("bert-base-cased-goemotions-original",
                          os.getcwd()+ "\\models\\bert-base-cased-goemotions-original")
acc = md.get_acc(os.getcwd()+'\\code\\data\\original\\test.tsv')

# evaluate the results : acc, square error, etc..
# model.config.id2label[idx]

print("Acc: {}".format(acc))

# 可以检查估计是否有偏？

#test ekman
md = Test_Eval_Model.Eval("bert-base-cased-goemotions-ekman",
                          os.getcwd()+ "\\models\\bert-base-cased-goemotions-ekman")
acc = md.get_acc(os.getcwd()+'\\code\\data\\ekman\\test.tsv')

print("Ekman Acc: {}".format(acc))

#test group
md = Test_Eval_Model.Eval("bert-base-cased-goemotions-group",
                          os.getcwd()+ "\\models\\bert-base-cased-goemotions-group")
acc = md.get_acc(os.getcwd()+'\\code\\data\\group\\test.tsv')

print("group Acc: {}".format(acc))