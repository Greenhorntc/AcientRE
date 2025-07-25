import pandas as pd
import os
import re
from sklearn.metrics import recall_score,precision_score,f1_score,classification_report,accuracy_score

from configs import mascotlabel2id
def LLMout_pair(floder):
    diafiles=os.listdir(floder)
    llmpair={}
    for datafile in diafiles:
        if 's1dia' in datafile:
            # print(datafile)
            dataname=datafile.split('.')
            # print(dataname)
            datafile=floder+datafile
            llmpair[dataname[0]]=datafile
        else:
            pass

    print(llmpair)
    return llmpair

def read_llmtxt(datafile):
    with open(file=datafile,encoding='utf-8') as f:
        lines=f.readlines()
        llmstr = ''.join(lines)
    return llmstr

def get_llm_pre(datafile):
    with open(datafile, 'r', encoding='utf-8') as file:
        # 逐行检查
        lines = file.read().splitlines()
    #直接取最后一行
    llmpre=lines[-1]
    print(llmpre)

    return llmpre

def extract_content(text):
    # 使用非贪婪匹配查找 ${...}$ 结构
    match = re.search(r'\$\{(.*?)\}\$', text)
    if match:
        return match.group(1)  # 返回标签内容

    return '手工查看'  # 没有匹配时返回None

def getdiapair(datapair):
    diasavefloder='data/fewshot/dialogs/' + datapair[0]+'/'
    diapairs=LLMout_pair(diasavefloder)
    print(diapairs)
    datadf = pd.read_excel(datapair[1])
    llmoutlst = []
    stage1out = []
    for index,row in datadf.iterrows():
        # print(row.id)
        # datakey='s2out'+str(row.id)
        datakey='s1dia'+str(row.id)
        if datakey in diapairs.keys():
            llmout = get_llm_pre(diapairs[datakey])
            print(llmout)
            label=extract_content(llmout)
            llmoutlst.append(label)
            # 所有对话输出
            llmout2 = read_llmtxt(diapairs[datakey])
            stage1out.append(llmout2)

        else:
            label = row['pre']
            stage1out.append('gan')
            llmoutlst.append(label)

    datadf['llm_out']=stage1out
    datadf['second_llm']=llmoutlst
    datadf['right'] = (datadf['label'] == datadf['second_llm']).astype(int)
    filename='data/fewshot/llmout/{}.xlsx'.format(datapair[0])
    datadf.to_excel(filename)
    print('over')



def getReport(rellabels,pre_label):
    acc=accuracy_score(rellabels, pre_label)
    print('测试acc')
    print(acc)
    Precision = precision_score(rellabels, pre_label, average='macro')
    print("测试Precision")
    print(Precision)
    recall = recall_score(rellabels, pre_label, average='macro')
    print("测试Recall")
    print(recall)
    f1 = f1_score(rellabels, pre_label, average='macro')
    print("测试F1")
    print(f1)
    # 返回模型的输出，用于集成学习计算概率。
    labels = list(range(0,32))
    # labels = list(range(0,25))
    print("report")
    print(classification_report(rellabels, pre_label, labels=labels,digits=5))


def checkllmout(llmout):
    labellst=list(mascotlabel2id.keys())
    if llmout not in labellst:
        llmout=-1
    else:
        llmout=mascotlabel2id[llmout]
    return llmout

def evaluate_file(savedfile):
    tempdataframe = pd.read_excel(savedfile)
    true_labels = tempdataframe['label'].tolist()
    llm_labels = tempdataframe['second_llm'].tolist()
    # llm_labels = tempdataframe['pre'].tolist()
    true_labels = [mascotlabel2id[x] for x in true_labels]
    llm_labels = [checkllmout(x) for x in llm_labels]
    # print(true_labels)
    # print(llm_labels)
    getReport(rellabels=true_labels, pre_label=llm_labels)

#
def evalutage_floder(datafloder):
    datafiles=os.listdir(datafloder)
    datafiles=[datafloder+x for x in datafiles]
    for datafile in datafiles:
        print('*******{}***********'.format(datafile))
        evaluate_file(datafile)
        print('*******')


if __name__ =='__main__':
    # fewshotfloder = 'data/fewshot/mergefiles/'
    # fewtestfiles = os.listdir(fewshotfloder)
    # dialogpath = [x.split('.xls')[0] for x in fewtestfiles]
    # fewtestfiles = [fewshotfloder + x for x in fewtestfiles]
    #
    # fewpairs = [[n, f] for n, f in zip(dialogpath, fewtestfiles)]
    # print(fewpairs)
    # for fewpair in fewpairs:
    #     # getdiapair(fewpair)

    # llmoutfloder='data/fewshot/llmout/'
    # evalutage_floder(llmoutfloder)


    '''
    单独进行评估, 0：dialogfloder 1：xlsx 需要组装的对话文件夹 和xlsx
    '''
    # gptpair=['combine-bert-ancient-chinese-mascot_full_gpt','data/fewshot/mergefiles/combine-bert-ancient-chinese-mascot_full.xlsx']
    # getdiapair(gptpair)
    evaluate_file('data/fewshot/llmout/combine-bert-ancient-chinese-mascot_full_gpt.xlsx')