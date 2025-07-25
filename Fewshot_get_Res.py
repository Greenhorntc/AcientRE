#实现逻辑，先进行错误分析。要记录不同的softmax的标签。用于二阶段大模型的分析
#预设条件 ： softmax的三个标签 可以命中真正的标签

import tensorflow as tf
from ReModelTraing import get_tf_dataset
from Remodel import  get_tokenizer_and_basemodel,getmodel
from configs import model2columns,dslabel,chislabel2id,cclabel2id
import numpy as np
from Re_evaluate import getReport
from datasets import Dataset,DatasetDict
import json
import os
import pandas as pd

"""1 配置ds """
# dsname cc or mascot
dsname = 'chis'
# dsname = 'cc'
dsselectlabel2id={'cc':cclabel2id,'chis':chislabel2id}
label2id=dsselectlabel2id[dsname]
id2label = {v: k for k, v in label2id.items()}

"""2 配置config"""
dsfloder = 'dataset/processed/re/' + dsname + '/all/'

#全量llm
# lms = ['bert-ancient-chinese', 'bertbase', 'guwenbert-base', 'guwenbert-large', 'sikubert', 'sikuroberta']

#根据需要选择
lms = ['bert-ancient-chinese',  'sikuroberta']

#应当存在以下模型：基线（什么都不加），加位置信息，加语义信息，加ema和什么都加
# clsmodel=['base','base_pos_resnet','base_semantic','base_resnet_EMA','base_pos_semantic_ema']

fewshots=['fewshot20','fewshot30','fewshot40','fewshot50','fewshot100','fewshot150','fewshot200']

clsmodel=['base_pos_resnet']


configs = []
for lm in lms:
    for fewshot in fewshots:
        for cls in clsmodel:
            #配置应包含 dsname。lm，clsmodel，modelsavedpath，savedresult，以及history
            ds=dsname
            savedmodel = 'savedmodel/refewshot/' + dsname + '/'+fewshot+'-' + lm+'-'+cls+'/'
            savedresult = 'savedResult/refewshot/' + dsname + '-' + lm + '-'+cls+'.txt'
            historypath= 'savedResult/re/' + dsname + '-' + lm + '-'+cls+'history.txt'
            presavepath='savedResult/refewshot/' + dsname + '/'+fewshot+'-' + lm+'-'+cls+'.xls'
            config = {'ds':dsname,"lm": lm, 'clsmodel':cls,"savemodelname": savedmodel, 'resultsavepath': savedresult,'histroy':historypath,'indexsave':presavepath}
            configs.append(config)

#获取所有的预测结果保存下来，预备进行第二轮大模型的增强。
def save_model_pres(y,presavepath):
    with open(presavepath, 'w', encoding='utf-8') as file:
        # 遍历列表中的每个元素，并将它们写入文件
        for category in y:
            file.write(f"{category}\n")
    print(presavepath+' saved over')

#因为给的是index+2，所以自然而然存在一定溢出。会出现预测13 14的label。所以需要裁剪下
def check_index(indexlst):
    tempindex=[]
    for preindex in indexlst:
        if preindex>=13:
            tempindex.append(0)
        else:
            tempindex.append(preindex)
    return tempindex
def save_pre_index(config,prelabels,indexlst,vaulelst):
    savepath = config['indexsave']
    # save_model_pres(prelabels,pressavepath)
    print(prelabels)
    print(indexlst)
    print(vaulelst)
    templst=[]
    for pre,preindex,prevalue in zip(prelabels,indexlst,vaulelst):
        preindex=check_index(preindex)
        preindex=[id2label[x] for x in preindex]
        templst.append([id2label[pre],preindex,prevalue])
    datadf = pd.DataFrame(templst, columns=['pre', 'index', 'possibilities'])
    # datadf.to_csv(path_or_buf=savepath,encoding='utf-8-sig')
    datadf.to_excel(savepath)
    print('saved '+savepath)

def evaluate_all_ds(config,model='test'):
    print(config)
    columns = model2columns[config['clsmodel']]
    print(columns)
    tokenizer, llm, data_collator = get_tokenizer_and_basemodel(config['lm'])
    train, val, test = get_tf_dataset(filepath=dsfloder, data_collator=data_collator, tokenizer=tokenizer,columns=columns)
    fcsize = dslabel[config['ds']]
    savedpath = config['savemodelname'] + '/model'
    clsmodel = getmodel(config['clsmodel'], llm, dp=0.7, fcsize=fcsize + 2)
    clsmodel.load_weights(savedpath)

    selected_data_dic={'train':train,'val':val,'test':test}
    sdata=selected_data_dic[model]

    predics = clsmodel.predict(sdata)
    #获取softmax概率，保留前三概率最高的标签，用于大模型验证。
    probabilities = tf.nn.softmax(predics, axis=-1)
    probabilities_np = probabilities.numpy()
    print(probabilities_np)

    #获取概率最大的标签作为预测结果
    prelabels = tf.math.argmax(probabilities, 1).numpy()

    """添加候选"""
    # 假设predics是你的模型预测输出，形状为[batch_size, num_classes]
    top_k_values, top_k_indices = tf.math.top_k(probabilities, k=4)
    top_k_indices_numpy = top_k_indices.numpy()
    top_k_values_numpy=top_k_values.numpy()
    modified_list = [[float(f'{num:.6f}') for num in sublist] for sublist in top_k_values_numpy.tolist()]

    # print(prelabels)
    # 获取测试集真实标签
    true_labels = []
    for input_data, label in sdata:
        true_labels.append(label.numpy())
    true_labels = np.concatenate(true_labels, axis=0)

    """
    """
    print(config['resultsavepath'])
    getReport(rellabels=true_labels, pre_label=prelabels, save=config['resultsavepath'])

    save_pre_index(config,prelabels,top_k_indices_numpy.tolist(),modified_list)
    return prelabels,true_labels


"""
先1 用config跑一下预测好的结果，然后和总数据集进行合并
"""
print(configs)
# possibilities=evaluate_all_ds(configs[0],model='test')
# for config in configs:
#     possibilities = evaluate_all_ds(config, model='test')

"""
2：得到pre的xlsx后合并整体测试集
"""
fewsavedfloder='savedResult/refewshot/{}/'.format(dsname)
fews=os.listdir(fewsavedfloder)
fews=[fewsavedfloder+x for x in fews]
basicfile='savedResult/chis_re_result_translation_fix.xlsx'
basicdf=pd.read_excel(basicfile)
basicdf_v2=basicdf.copy()
#2.1 合并

for few in fews:
    print(few)
    changecolomndf=pd.read_excel(few)
    basicdf_v2[["pre",'index_4','possibilities']]=changecolomndf[["pre",'index','possibilities']]
    basicdf_v2['right']=(basicdf_v2['pre'] == basicdf_v2['label']).astype(int)
    basicdf_v2.to_excel(few)

