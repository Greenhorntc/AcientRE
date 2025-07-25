import os
from datasets import DatasetDict
from ReModelTraing import *

dsname = 'chis'
dsfloder = 'dataset/processed/re/' + dsname + '/'
datasets=os.listdir(dsfloder)
dspaths=[dsfloder+x for x in datasets]
print(datasets)

llmpath={
    'bert-ancient-chinese': r"E:\Language_model\bert-ancient-chinese",
    'bertbase': r"E:\Language_model\bertbase",
    'guwenbert-base': r"E:\Language_model\guwenbert-base",
    'guwenbert-large': r"E:\Language_model\guwenbert-large",
    'sikubert': r"C:\Users\28205\PycharmProjects\LLMs\models\siku-bert",
    'sikuroberta': r"C:\Users\28205\PycharmProjects\LLMs\models\sikuroberta",
         }

lms = ['bertbase','guwenbert-base','bert-ancient-chinese', 'sikubert','sikuroberta']
cls='base_pos_resnet'
configs = []


for lm in lms:
    #去掉all
    for dataset in datasets[1:]:
        savedmodel = 'savedmodel/refewshot/' + dataset + '-' + lm + '-' + cls + '/model'
        savedresult = 'savedResult/refewshot/' + dataset + '-' + lm + '-' + cls + '.txt'
        historypath = 'savedResult/refewshot/' + dataset + '-' + lm + '-' + cls + 'history.txt'
        dspath=dsfloder+dataset
        config = {'ds':dsname,'dspah':dspath,"lm": lm, 'clsmodel':cls,"savemodelname": savedmodel, 'resultsavepath': savedresult,'histroy':historypath}
        configs.append(config)

# """
# 自定义模型保存全部保存的是模型权重而不是整个模型
# """
for config in configs:
    print(config)
    model_train(dspath=config['dspah'],config=config)
