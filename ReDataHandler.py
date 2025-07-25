"""
用于处理Re任务的数据集，两个开源数据
cc 和 mascot
"""
import json
import os
import ast
import opencc
import random
from collections import defaultdict, Counter
from configs import mascotlabel,mascotlabel2id,cclabel2id,chislabel2id
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets import Dataset, DatasetDict
import re
import random
# from TranslateUsingXunzi import re_data_process

class Re_data_handler():
    def __init__(self,):
        self.ccfloder='dataset/rawdata/c-clue/re/'
        self.mascot='dataset/rawdata/mascot/data/'
        self.CHisIEC='dataset/rawdata/CHisIEC-main/data/re/'

    def get_raw_mascot(self):
        mastrain=self.mascot+'train_acd.json'
        mastest=self.mascot+'test_acd.json'
        train_data = json.load(open(mastrain, "r"))
        test_data = json.load(open(mastest, "r"))
        return train_data,test_data

    def get_raw_cc(self):
        ccfiles=os.listdir(self.ccfloder)
        ccfiles=[self.ccfloder+x for x in ccfiles]
        print(ccfiles)
        trian=self.read_cc_json(ccfiles[2])
        val=self.read_cc_json(ccfiles[0])
        test=self.read_cc_json(ccfiles[1])
        return trian,val,test

    def get_raw_CHisIEC(self):
        train=self.CHisIEC+'coling_train.json'
        val=self.CHisIEC+'coling_train_dev.json'
        test=self.CHisIEC+'coling_test.json'
        train=json.load(open(train,'r',encoding='utf-8'))
        val=json.load(open(val,'r',encoding='utf-8'))
        test=json.load(open(test,'r',encoding='utf-8'))
        return train,val,test


    # 保存的鸭羹不是json，用文本格式读取
    def read_cc_json(self,file):
        lst=[]
        with open(file, 'r', encoding='utf8')as fp:
            lines=fp.readlines()
            for line in lines:
                data=ast.literal_eval(line)
                lst.append(data)
        print(len(lst))
        return lst

    '''
    数据dic格式 text： xx ,spo_list{ 'predicate':"关系，‘object_type’：“p”，subtype，object：xx，subject：xx}
    存在嵌套实体 要分割成两条句子
    '''
    def change_mascot_data(self,datadic):
        temp = []
        for datakey in datadic:
            for data in datadic[datakey]:
                jsondata=self.get_detail_mascot(datakey,data)
                temp+=jsondata
        #删除重复数据
        deldata=self.mascot_data_del(temp)
        return deldata

    # change cc 到统一格式,ccdata 全部为单个对应
    def change_cc_data(self, datalst):
        changeddata = []
        for datadic in datalst:
            text = datadic['text']
            relation = datadic['spo_list']['predicate']
            # 用主体ner来当做头实体
            data = {
                "sentence": text,
                "entities": [
                    {'entity1': datadic['spo_list']['subject'], 'type': datadic['spo_list']['subject_type']},
                    {'entity2': datadic['spo_list']['object'], 'type': datadic['spo_list']['object_type']}],
                "relations": [{"entity1": datadic['spo_list']['subject'], "entity2": datadic['spo_list']['object'],
                               "relation": relation}]
            }
            changeddata.append(data)
        return changeddata

    def change_CHisIEC_data(self,chisdatalist):
        #1:分解数据
        #2：标签扩展 标签扩展的主要目的是应对下游任务，无法确定头尾实体的时候，可能会产生e2 e1颠倒的问题
        changeddata = []

        for datadic in chisdatalist:
            text = datadic['tokens']
            ners=datadic['entities']
            relations=datadic['relations']

            #3 分两种情况，1 存在关系 和 2 不存在关系的，有的标注不存在关系
            relen=len(relations)
            if relen==0:
                if len(ners)==0:
                    pass
                else:
                    # print(text)
                    # print("不存在关系")
                    # print(ners)
                    subner,objner=self.random_select(ners,2)
                    subner['link']='SUB'
                    objner['link'] = 'OBJ'
                    marked_text = self.mark_entities_chi(text, [subner, objner])
                    changeddata.append([marked_text,text,chislabel2id['不存在关系'],'翻译暂时不存在'])
            else:
                for relation in relations:
                    # print(relation)
                    subner=ners[relation['head']]
                    subner['link']='SUB'
                    objner=ners[relation['tail']]
                    objner['link'] = 'OBJ'
                    marked_text = self.mark_entities_chi(text, [subner,objner])
                    marked_text = "[CLS]{}[SEP]".format(marked_text)
                    changeddata.append([marked_text,text,chislabel2id[relation['type']],'翻译暂时不存在'])
        return changeddata


    def random_select(self,selectlist,n):
        sublist=random.sample(selectlist,n)
        return sublist

    def get_detail_mascot(self, key, data):

        all_data = []
        text = ''.join(data['tokens'])
        text = self.traditional_to_simplified(text)
        headners = self.traditional_to_simplified(data['h'][0].replace(' ', '')).split(':')
        tailners = self.traditional_to_simplified(data['t'][0].replace(' ', '')).split(':')
        entities = []
        relations = []
        # for headner in headners:
        #     entities.append({"entity": headner, "type": mascotlabel[key][0]})
        #     for tailner in tailners:
        #         entities.append({"entity": tailner, "type": mascotlabel[key][1]})
        #         relations.append({"entity1": headner, "entity2": tailner, "relation": key})
        entity_counter = 1

        for headner in headners:
            head_entity_label = "entity{}".format(entity_counter)
            entities.append({head_entity_label: headner, "type": mascotlabel[key][0]})
            entity_counter += 1

            for tailner in tailners:
                tail_entity_label = "entity{}".format(entity_counter)
                entities.append({tail_entity_label: tailner, "type": mascotlabel[key][1]})
                relations.append({"entity1": headner, "entity2": tailner, "relation": key})
                entity_counter += 1
        entities=self.remove_duplicate_entities(entities)

        data_entry = {
            "sentence": text,
            "entities": entities,
            "relations": relations
        }
        all_data.append(data_entry)

        return all_data

    # mascot数据存在一定冗余，多条数据重复出现的情况。根据 text文本进行去重
    def remove_duplicate_entities(self,entities):
        seen = set()
        unique_entities = []

        for entity in entities:
            entity_name = list(entity.values())[0]
            if entity_name not in seen:
                seen.add(entity_name)
                unique_entities.append(entity)

        return unique_entities

    #mascot数据存在一定冗余，多条数据重复出现的情况。根据 text文本进行去重
    def mascot_data_del(self,dlist):
        # 使用集合用于去重，并保持顺序
        seen_texts = set()
        unique_list = []
        for item in dlist:
            text = item['sentence']
            if text not in seen_texts:
                seen_texts.add(text)
                unique_list.append(item)
        return unique_list


    # 简繁体转换
    def traditional_to_simplified(self, traditional_text):
        cc = opencc.OpenCC('t2s.json')  # 加载繁体字转简体字的配置文件
        simplified_text = cc.convert(traditional_text)
        return simplified_text



    #将json数据转换成训练数据,
    def create_ds_data(self,alldata,datatype):
        tempall=[]
        #按照关系类别进行数据采样，mascot需要进行处理,另外引入符号
        print("ordata to traning sample")

        for data in alldata:
            text=data["sentence"]
            rels = data['relations']
            entities = data['entities']
            # 数据格式 ner1 ner2 reltion
            for nerpair in rels:
                if datatype == "mascot":
                    label2id=mascotlabel2id
                elif datatype=="cc":
                    label2id = cclabel2id
                else:
                    raise ValueError('Get wrong input')
                rel=label2id[nerpair['relation']]
                ner1 = nerpair['entity1']
                ner2 = nerpair['entity2']
                #根据ner回去找nertype
                nertype1=self.find_entity_type(ner1,entities)
                nertype2=self.find_entity_type(ner2,entities)
                #加入标签符号
                entitiesdic = [
                    {'entity': ner1, 'type': "SUB_"+nertype1},
                    {'entity': ner2, 'type': 'OBJ_'+nertype2}
                ]
                marked=self.mark_entities(text,entitiesdic)
                tempall.append([marked,text,rel])
        return tempall

    # #加入标记，
    # def mark_entities(self,sentence, entities):
    #     # Sort entities by their starting index in reverse order
    #     # sorted_entities = sorted(entities, key=lambda x: sentence.find(x['entity']), reverse=True)
    #
    #     sorted_entities = sorted(entities, key=lambda x: len(x['entity']), reverse=True)
    #     print('mark')
    #     print(sentence)
    #     print(sorted_entities)
    #     #先标记长的实体
    #     first_entity=sorted_entities[0]
    #     first_entity_text = first_entity['entity']
    #     first_entity_type = first_entity['type']
    #     marked_sentence = sentence.replace(
    #         first_entity_text, f"[{first_entity_type}]{first_entity_text}[/{first_entity_type}]"
    #     )
    #     print(marked_sentence)
    #
    #     second_entity = sorted_entities[1]
    #     second_entity_text = second_entity['entity']
    #     second_entity_type = second_entity['type']
    #
    #     # 使用 finditer 查找所有匹配
    #     for match in re.finditer(re.escape(second_entity_text), marked_sentence):
    #         start, end = match.span()
    #         print(start)
    #         print(end)
    #         # 检查第二个实体是否在第一个实体标记中
    #         print(len(marked_sentence))
    #         '''
    #         其实就是没考虑end就是最后一个实体的情况
    #         '''
    #         print(marked_sentence[start - 1])
    #         print(marked_sentence[end])
    #         if marked_sentence[start - 1] != '[' and marked_sentence[end] != ']':
    #             # 标记第二个实体
    #             marked_sentence = marked_sentence[:start] + f"[{second_entity_type}]{second_entity_text}[/{second_entity_type}]" + marked_sentence[end:]
    #             break
    #     # print(marked_sentence)
    #     return "[CLS] {} [SEP]".format(marked_sentence)

    def mark_entities(self,sentence, entities):
        # Sort entities by their starting index in reverse order
        # sorted_entities = sorted(entities, key=lambda x: sentence.find(x['entity']), reverse=True)

        sorted_entities = sorted(entities, key=lambda x: len(x['entity']), reverse=True)
        print('mark')
        print(sentence)
        print(sorted_entities)
        #先标记长的实体
        first_entity=sorted_entities[0]
        first_entity_text = first_entity['entity']
        first_entity_type = first_entity['type']
        marked_sentence = sentence.replace(
            first_entity_text, f"[{first_entity_type}]{first_entity_text}[/{first_entity_type}]"
        )
        print(marked_sentence)

        second_entity = sorted_entities[1]
        second_entity_text = second_entity['entity']
        second_entity_type = second_entity['type']

        pattern = re.compile(re.escape(second_entity_text))
        for match in pattern.finditer(marked_sentence):
            start, end = match.span()
            if not (marked_sentence[start - 1:start + len(second_entity_text)].startswith('[') and marked_sentence[end - 1:end + 1].endswith( ']')):
                marked_sentence = marked_sentence[:start] + f"[{second_entity_type}]{second_entity_text}[/{second_entity_type}]" + marked_sentence[end:]
                break
        print(marked_sentence)
        return "[CLS] {} [SEP]".format(marked_sentence)

    def mark_entities_chi(self,text, entities):
        # Sort entities by start index in reverse order to avoid messing up indices
        entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        # print(entities)
        # Mark each entity in the text
        for entity in entities:
            start = entity['start']
            end = entity['end']
            span = entity['span']
            entity_type = entity['type']
            link=entity['link']
            # Create the tags
            start_tag = f"<{link}_{entity_type}>{span}</{link}_{entity_type}>"

            # Replace the span in text with the tagged version
            text = text[:start] + start_tag + text[end:]

        return text


    def find_entity_type(self,entity_name,entities):
        for entity in entities:
            if entity_name in entity.values():
                return entity['type']
        return None

    def sample_data_by_rel(self,samplellist):
        #根据关系进行样本的采样,使用skllearn的分层抽样实现。
        print("sample by relationship")
        # 转换为 DataFrame
        df = pd.DataFrame(samplellist, columns=['text', 'originaltext', 'label', 'translation'])
        # 分层抽样，按标签分层
        train, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

        val, test = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=42)
        # 输出结果
        print("训练集：")
        print(len(train))
        print("\n验证集：")
        print(len(val))
        print("\n测试集：")
        print(len(test))
        return train,val,test

    def mascot_data_temp(self):
        train, test =self.get_raw_mascot()
        train = self.change_mascot_data(train)
        test = self.change_mascot_data(test)
        all = train + test
        all = self.create_ds_data(all,datatype='mascot')
        self.data_to_temp(all, dsname='mascot')

    def cc_data_temp(self):
        train,val,test=self.get_raw_cc()
        train=self.change_cc_data(train)
        val=self.change_cc_data(val)
        test=self.change_cc_data(test)
        train=self.create_ds_data(train,datatype='cc')
        val=self.create_ds_data(val,datatype='cc')
        test=self.create_ds_data(test,datatype='cc')
        dslst=[train,val,test]
        #temp [1,2,3] 1: [marked,rel]
        self.data_to_temp(dslst,dsname='cc')

    def CHisIEC_data_temp(self):
        train,val,test=self.get_raw_CHisIEC()

        train=self.change_CHisIEC_data(train)
        val=self.change_CHisIEC_data(val)
        test=self.change_CHisIEC_data(test)
        dslst = [train, val, test]
        self.data_to_temp(dslst, dsname='chis')



    def read_data_temp(self, files):
        print(files)
        if len(files) == 1:
            mascot = json.load(open(files[0], "r", encoding='utf-8'))
            return mascot
        else:
            train = json.load(open(files[0], "r", encoding='utf-8'))
            val = json.load(open(files[1], "r", encoding='utf-8'))
            test = json.load(open(files[2], "r", encoding='utf-8'))
            return train, val, test

    """
    1: retemp目录保存分割好的数据，第二次加载直接从temp加载
    2：新增特征也修改的temp数据
    3：re.xx_deal也先检查temp存在不存在，如果存在直接用
    4：fewshot也从temp里面读取
    """
    def data_to_temp(self,datalist,dsname):
        print("data to temp for add features ")
        retempfloder='dataset/processed/re/retemp/'+dsname+'/'
        if not os.path.exists(retempfloder):
            os.makedirs(retempfloder)

        if dsname=='cc' or "chis":
            trainfile=retempfloder+'trian.json'
            valfile=retempfloder+'val.json'
            testfile=retempfloder+'test.json'
            self.sava_data_to_json(datalist[0],trainfile)
            self.sava_data_to_json(datalist[1],valfile)
            self.sava_data_to_json(datalist[2],testfile)
        elif dsname=='mascot':
            fname=retempfloder+'all.json'
            self.sava_data_to_json(datalist, fname)
        else:
            raise ValueError('input ？？？')


    def sava_data_to_json(self,data,file):
        # 写入JSON数据到文件
        with open(file, 'w',encoding='utf-8') as f:
            json.dump(data, f,ensure_ascii=False,indent=4)
            print(file+"   "+ "saved")

    def ds_to_disk(self,train,val,test,name):
        print('Saving to disk')

        # 将数据划分为训练集、验证集和测试集
        dataset_dict = DatasetDict({
            "train": Dataset.from_dict(train),
            "validation": Dataset.from_dict(val),
            "test": Dataset.from_dict(test)
        })

        # 保存到硬盘
        dataset_dict.save_to_disk("dataset/processed/re/"+name)
        print('saved over')



    def data_deal(self, dsname,fewshot=False):
        # 先看有没有json数据，有就直接读，没有就执行temp
        retempfloder = 'dataset/processed/re/retemp/' + dsname + '/'
        files = os.listdir(retempfloder)
        for file in files:
            if file.endswith('.json'):
                print('temp files has been created')
                break
            else:
                if dsname == 'mascot':
                    self.mascot_data_temp()
                    break
                elif dsname == "cc":
                    self.cc_data_temp()
                    break
                elif dsname=='chis':
                    self.CHisIEC_data_temp()
                else:
                    raise ValueError('error input')

        files = [retempfloder + x for x in files]
        print(files)
        #mascot 数据集随机采样
        if dsname == 'mascot':
            data = self.read_data_temp(files)
            train, val, test = self.sample_data_by_rel(data)
        #cc和chis 直接按文件读取
        else:
            #注意：这里文件名 read 完以后 要注意下 不知道为啥有时候会
            test,train, val = self.read_data_temp(files)

        if fewshot == True:
            print('进行fewshot，每个类别按照 20 30 40 50 100 150 200 进行采样，不够的就按照该类数据的最大数量进行采样')
            samplelist=[20,30,40,50,100,150,200]
            for n in samplelist:
                fewtrain=self.do_train_fewshot(train,sample=n)
                print(len(fewtrain))
                fewname=dsname+'/fewshot{}'.format(str(n))
                print(fewname)
                fewtrain = pd.DataFrame(fewtrain, columns=['text', 'originaltext', 'label', 'translation'])
                val = pd.DataFrame(val, columns=['text', 'originaltext', 'label', 'translation'])
                test = pd.DataFrame(test, columns=['text', 'originaltext', 'label', 'translation'])
                self.ds_to_disk(fewtrain,val,test,fewname)
        dssave=dsname+'/all'
        train = pd.DataFrame(train, columns=['text', 'originaltext', 'label', 'translation'])
        val = pd.DataFrame(val, columns=['text', 'originaltext', 'label', 'translation'])
        test = pd.DataFrame(test, columns=['text', 'originaltext', 'label', 'translation'])
        self.ds_to_disk(train,val,test,dssave)

    def do_train_fewshot(self,train,sample):
        # 统计每个标签的数量
        label_counts = Counter([sample[2] for sample in train])
        # print(label_counts)
        n_per_label = sample
        # 初始化一个空的 few-shot 数据集
        few_shot_dataset = []
        # 遍历每个标签
        for label, count in label_counts.items():
            # 计算可以从该标签中抽取的最大样本数量
            max_n = min(n_per_label, count)
            # 随机选择 max_n 个该标签的样本
            selected_indices = random.sample(
                [i for i, (text, originaltext, lbl, translation) in enumerate(train) if lbl == label],
                max_n
            )

            # 将选中的样本添加到 few-shot 数据集中
            few_shot_dataset.extend([train[i] for i in selected_indices])
        # 注意：few_shot_dataset 的长度可能小于 n_per_label * 标签数量，
        # 因为有些标签的样本数量可能少于 n_per_label。
        return few_shot_dataset

if __name__=="__main__":
    rehandler=Re_data_handler()
    #1 单独创建temp
    # rehandler.cc_data_temp()
    # rehandler.mascot_data_temp()
    #2 翻译
    # re_data_process()
    #3 datadeal
    # rehandler.data_deal(dsname='cc')
    # rehandler.data_deal(dsname='mascot')
    rehandler.data_deal(dsname='chis',fewshot=True)