from datasets import DatasetDict
from LLMRe.UsingDeepSeek import *
from LLMRe.configs import mascotlabel2id
import re
import os
mascotid2label = {v: k for k, v in mascotlabel2id.items()}

mascotds='../dataset/processed/re/mascot/mascot_full/'
ds=DatasetDict.load_from_disk(mascotds)

def read_llm_file(llmfile):
    tempindex=0
    with open(llmfile,encoding='utf-8') as f:
        lines = f.readlines()
        for i  in range(len(lines)):
            if '</think>' in lines[i]:
                tempindex=i
        lines=lines[tempindex+1:]
        llmstr = ''.join(lines)
    return llmstr

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

#定制dialog，然后询问模型。
def getRedes_LLm(data):
    label = data['label']
    prompt='你非常了解关系抽取任务，并能够从给定的信息中总结关系的定义。'
    dialog = '完整信息如下：\n'
    text = '文言文为：{{{}}}\n'.format(data['originaltext'])
    nerds=get_ner_des(data['text'])
    # nerdes, headner, tailner, htype, tailtype = get_ner( data['text'])
    translation = '句子的翻译为：{{{}}}\n'.format(data['translation'])
    redes="实体1和实体2的关系是{}\n".format(mascotid2label[label])
    querry="请根据给定的完整信息，帮我总结关系{}的定义。\n".format(mascotid2label[label])
    llminput=dialog+text+translation+nerds+redes+querry
    print(llminput)
    s1messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": llminput},
    ]
    # print(s1messages)
    return s1messages

def get_llm_dia(label):
    labelfloder='redes/record/'
    lfiles=os.listdir(labelfloder)
    lfiles=[x for x in lfiles if label in x]
    lfiles=[labelfloder+x for x in lfiles]
    # print(lfiles)
    llmstrs=[]
    for i in range(len(lfiles)):
        llmstr=read_llm_file(lfiles[i])
        desone="{}定义{}：{{\n{}\n}},\n".format(label,i+1,llmstr)
        llmstrs.append(desone)
    allllmout=''.join(llmstrs)
    return allllmout

def s1message(label):
    prompt='你非常善于总结实体关系定义，将给定的多组的关系定义进行总结统一。'
    baseredes='实体关系为：{{{}}}\n'.format(label)
    diastart='提供的多组{}关系定义如下：\n'.format(label)
    relabeldes=get_llm_dia(label)
    diacot = '\n-----------------分隔--------------\n注意事项：\n1. 不要遗漏每一条定义给出的关键内容。\n2. 定义之间可能存在矛盾之处，着重处理矛盾保持整体定义统一。\n3. 要概括标签定义的方方面面。\n4,在基础定义进行修改。'

    s1input=diastart+baseredes+relabeldes+diacot
    print(s1input)
    s1messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": s1input},
    ]
    return s1messages


def save_refile(label,llmout,id):
    copyfile='redes/record/{}{}.txt'.format(label,id)
    with open(copyfile, 'w', encoding='utf-8') as f:
        f.write(llmout)

def get_llm_re(ds):
    id=0
    for data in ds['train']:
        label = data['label']
        d_m=getRedes_LLm(data)
        llmout=ask_deepseek_8b_messages(d_m)
        print(llmout)
        save_refile(mascotid2label[label],llmout,id)
        id=id+1

def summery_files(datalabel):
    #1： 无须创建，直接生成。
    s1m = s1message(datalabel)
    llmout = ask_deepseek_671B_messages(s1m)
    # llmout=ask_deepseek_8b_messages(s1m)

    print(llmout)
    copyfile = 'redes/{}.txt'.format(datalabel)
    with open(copyfile, 'w', encoding='utf-8') as f:
        f.write(llmout)

    print('over')

if __name__ == '__main__':
    print(mascotlabel2id)
    labellst=list(mascotlabel2id.keys())

    get_llm_re(ds)

    # for laebel in labellst:
    #     summery_files(laebel)



# redesinput=get_llm_dia(labellst[0])
# print(redesinput)