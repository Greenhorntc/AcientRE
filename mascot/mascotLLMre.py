import os
import pandas as pd
from LLMRe.UsingDeepSeek import *
from LLMRe.QueryByChatGpt import re_gpt4o_singleQA
import time
from LLMRe.mascot.mascotRedes import mascotdic
import re
import ast
from configs import mascotlabel
#fewshots
fewshotfloder='data/fewshot/mergefiles/'

fewtestfiles=os.listdir(fewshotfloder)
dialogpath=[x.split('.xls')[0] for x in fewtestfiles]
fewtestfiles=[fewshotfloder+x for x in fewtestfiles]
fewpairs=[[n,f] for n,f in zip(dialogpath,fewtestfiles)]
print(fewpairs)

#读取实体
def extract_entities(text):
    """
    从给定的文本中提取实体。

    参数:
        text (str): 包含实体标记的文本。

    返回:
        list: 包含提取的实体元组列表，每个元组格式为 (entity_type, entity_content)。
    """
    # 定义正则表达式模式以匹配实体
    pattern = re.compile(
        r'\[(SUB|OBJ)_([^\]]+)\](.*?)\[/\1_\2\]',
        flags=re.DOTALL
    )

    matches = pattern.finditer(text)
    entities = []
    for match in matches:
        entity_type = match.group(2).strip()  # 提取并清理实体类型
        entity_content = match.group(3).strip()  # 提取并清理实体内容
        entities.append((entity_type, entity_content))

    return entities

def get_ner_des(sentence):

    pattern = r'\[CLS\]|\[SEP\]'
    cleaned_text = re.sub(pattern, '', sentence)
    entities=extract_entities(cleaned_text)
    nerdes="实体信息：{{实体1：{},实体2：{}}}\n".format(entities[0][1],entities[1][1])
    return nerdes


def get_re_des_singalQA(prelist):
    redes_singaQA=[]
    for i in range(len(prelist)):
        reinformation = mascotdic[prelist[i]].rstrip()
        if i!=len(prelist)-1:
            # prestr = '{{[{}]:{}。\n}},'.format(prelist[i], reinformation)
            prestr = '{}：{{{}}},\n'.format(prelist[i], reinformation)

            redes_singaQA.append(prestr)
        else:
            prestr = '{}：{{{}}}'.format(prelist[i], reinformation)
            # fewshotdes = getfewshot_dig(prelist[i])
            redes_singaQA.append(prestr)
            # redes_singaQA.append(fewshotdes)
    redesstr='给定的候选实体关系：{\n'+'\n'.join(redes_singaQA)+'\n}'
    # print(redesstr)
    return redesstr

def save_temp_data(res, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        # 将字符串写入文件
        file.write(res)

def one_stage_save(dataid,s1messages,diapath):
    diasavefloder = 'data/fewshot/dialogs/' + diapath
    if not os.path.exists(diasavefloder):
        os.makedirs(diasavefloder)
    templlm = []
    for llm in s1messages:
        templlm.append(llm['content'])
    finalsavestr = '\n'.join(templlm)
    diafile = diasavefloder + '/s1dia{}.txt'.format(dataid)
    save_temp_data(finalsavestr, filename=diafile)
    print('saved over')

#采用单论对话
def combine_data_2_sentence_in_one_stage(data):
    prompts1='你是一名精通文言文实体关系分析的专家，你的任务是：从给定的候选关系中选择最有可能的候选项来描述两个实体之前的关系。\n'
    dialog='完整信息如下：\n'
    pattern = r'\[CLS\]|\[SEP\]'
    cleaned_text = re.sub(pattern, '', data.text_with_tag)
    text = '文言文为：{{{}}}\n'.format(cleaned_text)
    nerdes = get_ner_des(cleaned_text)
    translation = '文言文的翻译为：{{{}}}\n'.format(data.tanslation)
    index_5 = ast.literal_eval(data.index_4)
    redes=get_re_des_singalQA(index_5)

    cot = '\n-----------------分隔--------------\n注意事项：\n1. 参考文言文的翻译。\n2. 参考候选实体关系的定义。\n3. 利用给定的实体信息。\n' \
          '\n请按以下步骤选择实体间最有可能的关系。\n1：分别判断两个实体之间的关系是候选关系的可能性。\n2：最终统筹判断，给出最有可能的关系。\n3：请不要给出候选关系以外的答案！\n'
    end='在回答的最后一行用${实体关系}$标记最终输出。\n'
    # # s1llminput = dialog + text + nerdes + translation +redes+fewdia+ cot
    s1llminput = dialog + text + translation +nerdes+redes + cot+end

    print(prompts1+s1llminput)
    s1messages= [
        {"role": "system", "content": prompts1},
        {"role": "user", "content": s1llminput},
    ]
    # print(s1messages)
    return s1messages

def get_Answer_from_llm_singleQA(data,diapath):

    s1fun=combine_data_2_sentence_in_one_stage
    print('-------------LLm stage1---------------')
    s1messages=s1fun(data)
    print('\n')
    s1out =ask_deepseek_671B_messages(s1messages)
    s1messages.append({'role': 'assistant', 'content': s1out})
    #s1 messageds 组装第二轮对话
    print(s1out)
    time.sleep(0.1)

    # saveLLm_out(data,allmessages,s2out=s2out,diapath=diapath)

    one_stage_save(data.id,s1messages,diapath)
    print('-------------Over---------------')


def get_Re_onestage_mainstream(testdf,dialogname):
    """修改保存逻辑，每条数据都保存"""
    for index,row in testdf.iterrows():
            get_Answer_from_llm_singleQA(row,dialogname)


def create_chatgpt_messages(data):
    prompt = '你是一个语义抽取工具，你的任务是识别实体间的关系。\n'
    mascotrelabel=list(mascotlabel.keys())
    redes="关系列表为：{{{}}}。\n".format(','.join(mascotrelabel))
    dialog = '完整信息如下：\n'
    pattern = r'\[CLS\]|\[SEP\]'
    cleaned_text = re.sub(pattern, '', data.text_with_tag)
    text = '文言文为：{{{}}}\n'.format(cleaned_text)
    nerdes = get_ner_des(cleaned_text)
    translation = '文言文的翻译为：{{{}}}\n'.format(data.tanslation)
    diainfo = '请在回答的最后一行输出形式为：${实体关系}$。\n'

    llmquery = dialog+text + nerdes + translation+redes+diainfo
    print(llmquery)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": llmquery},
    ]
    return messages

#新增gpt 对话,不分任何类别，全部都标签直接询问大模型。
def Chat_GPT_dialog(dataframe,diasavepath):
    for index,row in dataframe.iterrows():
        gptm=create_chatgpt_messages(row)
        # print(row.text_with_tag)
        llmout=re_gpt4o_singleQA(gptm)
        print(llmout)
        gptm.append({'role': 'assistant', 'content': llmout})
        one_stage_save(row.id, s1messages=gptm, diapath=diasavepath)
        time.sleep(0.2)


if __name__ == '__main__':


    print("start")
    # for fewpair in fewpairs:
    #     taget=fewpair[0]
    #     testdf=pd.read_excel(fewpair[1])
    #     get_Re_onestage_mainstream(testdf, taget)
    # get_Answer_from_multi_QA(testdf,taget[0])

    """
    单独
    """
    print(fewpairs[4])
    taget =fewpairs[4]

    testdf = pd.read_excel(taget[1])
    diasave=taget[0]+'_gpt'
    Chat_GPT_dialog(dataframe=testdf,diasavepath=diasave)