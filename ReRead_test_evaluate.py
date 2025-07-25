#用于读取保存的数据再进行评估，bert ancient large经常超显存
import tensorflow as tf
from ReModelTraing import get_tf_dataset
from Remodel import  get_tokenizer_and_basemodel,getmodel
from configs import model2columns,dslabel,chislabel2id
import numpy as np
from Re_evaluate import getReport
from datasets import Dataset,DatasetDict
import json

# dsname cc or mascot
# dsname = 'cc'

dsname = 'chis'
dsfloder = 'dataset/processed/re/' + dsname + '/all/'

# lms = ['bert-ancient-chinese', 'bertbase', 'guwenbert-base', 'sikubert', 'sikuroberta']

lms = ['bert-ancient-chinese']
#应当存在以下模型：基线（什么都不加），加位置信息，加语义信息，加ema和什么都加
# clsmodel=['base','base_pos_resnet','base_semantic','base_resnet_EMA','base_pos_semantic_ema']
#已知base
clsmodel=['base']
configs = []
for lm in lms:
    for cls in clsmodel:
        #配置应包含 dsname。lm，clsmodel，modelsavedpath，savedresult，以及history
        ds=dsname
        savedmodel = 'savedmodel/re/' + dsname + '-' + lm+'-'+cls+'/'
        savedresult = 'savedResult/re/' + dsname + '-' + lm + '-'+cls+'.txt'
        historypath= 'savedResult/re/' + dsname + '-' + lm + '-'+cls+'history.txt'
        config = {'ds':dsname,"lm": lm, 'clsmodel':cls,"savemodelname": savedmodel, 'resultsavepath': savedresult,'histroy':historypath}
        configs.append(config)

#获取所有的预测结果保存下来，进行第二轮大模型的增强。
def save_model_pres(y,presavepath):
    with open(presavepath, 'w', encoding='utf-8') as file:
        # 遍历列表中的每个元素，并将它们写入文件
        for category in y:
            file.write(f"{category}\n")
    print(presavepath+' saved over')
def evaluate_test_ds(config):
    print(config)
    columns = model2columns[config['clsmodel']]
    print(columns)
    tokenizer, llm, data_collator = get_tokenizer_and_basemodel(config['lm'])
    train, val, test = get_tf_dataset(filepath=dsfloder, data_collator=data_collator, tokenizer=tokenizer,columns=columns)
    fcsize = dslabel[config['ds']]
    savedpath = config['savemodelname'] + '/model'

    clsmodel = getmodel(config['clsmodel'], llm, dp=0.7, fcsize=fcsize + 2)
    clsmodel.load_weights(savedpath)

    predics = clsmodel.predict(test)
    prelabels = tf.math.argmax(predics, 1).numpy()
    # print(prelabels)

    # 获取测试集真实标签
    true_labels = []
    for input_data, label in test:
        true_labels.append(label.numpy())
    true_labels = np.concatenate(true_labels, axis=0)
    pressavepath = 'savedResult/' + config['lm'] + '_' + config['clsmodel'] + "pre.txt"
    save_model_pres(prelabels,pressavepath)

    return prelabels,true_labels

# for config in configs:
#     pres,y=evaluate_test_ds(config)
#     getReport(rellabels=y, pre_label=pres, save=config['resultsavepath'])

""" 单个评估"""
#1 bert-acient 在chis
#2 sikuroberta 在cc上较好 sikuroberta+ ner_mark+pos +resnet  configs[4]

print(configs)
pres, y=evaluate_test_ds(configs[0])
getReport(rellabels=y, pre_label=pres, save=configs[0]['resultsavepath'])


# allds= DatasetDict.load_from_disk(dsfloder)
#
# #查看具体类别下，真正样本 的预测 和需要查看的类别
# def check_details(config):
#     errordic={}
#     pres, y=evaluate_test_ds(config)
#     preslist=pres.tolist()
#     id2label={v: k for k, v in chislabel2id.items()}
#     for y_ds,pre,text,ortext in zip(allds['test']["label"],preslist,allds['test']["text"],allds['test']['ortext']):
#         rellabel = id2label[y_ds]
#         prelabel = id2label[pre]
#
#         #保存错误信息
#         if y_ds!=pre:
#             if rellabel not in errordic:
#                 errordic[rellabel]=[[prelabel,text]]
#             else:
#                 errordic[rellabel].append([prelabel,text])
#     errorsavepath = 'savedResult/'+config['lm'] +'_'+ config['clsmodel'] + ".json"
#
#     with open(errorsavepath, 'w', encoding='utf-8') as json_file:
#         json.dump(errordic, json_file, ensure_ascii=False, indent=4)
#     print(errorsavepath+'.........saved over')
#     return errordic
#
# # for con in configs:
# #     details=check_details(config=con)
# #     print(details)
#
# details=check_details(config=configs[4])


