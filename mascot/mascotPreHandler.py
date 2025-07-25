import pandas as pd
from datasets import DatasetDict
from configs import mascotlabel2id
import ast
import os

id2label = {v: k for k, v in mascotlabel2id.items()}


mascotfloder='data/fewshot/unmerge/'
#1:将pre和test进行混合
mffiles=os.listdir(mascotfloder)
mascotfews=[mascotfloder+x for x in mffiles]

# mascotprefile='data/mascot/mascot-bert-ancient-chinese.xlsx'
# predf=pd.read_excel(mascotprefile,index_col=0)
# print(predf.head(4))

mascotds='../dataset/processed/re/mascot/mascot_full'
ds=DatasetDict.load_from_disk(mascotds)
datalst=[]
i=0
for data in ds['test']:
    # print(data)
    i=i+1
    tempdic={'id':i,'text_with_tag':data['text'],'text':data['originaltext'],'tanslation':data['translation'],'label':id2label[data['label']]}
    datalst.append(tempdic)

datadf=pd.DataFrame(datalst)


def index_4(labellst,choicelst):
    index_4=[]
    for y,choices in zip(labellst,choicelst):
        print(y)
        print(type(choices))
        choiceslst=ast.literal_eval(choices)
        print(choiceslst)
        if y in choices:
            index_4.append(1)
        else:
            index_4.append(0)
    return index_4

def combine_data(predf,fewshotname):
    alldf = pd.concat([datadf, predf], axis=1)
    alldf.drop(alldf.index[-1], inplace=True)
    print(alldf.columns)

    alldf=alldf.rename(columns={'index': 'index_5'})
    alldf['right_pre'] = (alldf['y'] == alldf['pre']).astype(int)
    index_4_lst=index_4(alldf['y'].tolist(),alldf['index'].tolist())
    alldf['index_4_right']=index_4_lst
    alldf.to_excel(fewshotname)
    print('over')

for  mascotfew,fewshot in zip(mascotfews,mffiles):
    # print(mascotfew)
    print(fewshot.split('.xlsx')[0])
    fewsavename='data/fewshot/mergefiles/combine-{}.xlsx'.format(fewshot.split('.xlsx')[0])
    predf=pd.read_excel(mascotfew,index_col=0)
    combine_data(predf,fewshotname=fewsavename)