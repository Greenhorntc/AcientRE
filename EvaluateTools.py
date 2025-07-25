import re
import ast
import json
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


# #使用模型预测测试集合 采用json数据集合进行预测
def readfile(file):
    with open(file, 'r', encoding='utf-8') as file:
        data = []
        for line in file.readlines():
            # print(line)
            dic = json.loads(line)
            data.append(dic)
    return data

def convert_to_bio(sentence):
    # 去除非中文标签
    sentence = re.sub(r'<[^>]+>', '', sentence)
    # 分词
    words = sentence.split(' ')
    bio_tags = []
    for word in words:
        if '/' in word:
            word, pos = word.split('/', 1)
            # print(word+pos)
            tag = "O"
            if pos == 'nr':
                tag = "B-PER"
                bio_tags.extend([tag] + ["I-PER"] * (len(word) - 1))
            elif pos == 'ns':
                tag = "B-LOC"
                bio_tags.extend([tag] + ["I-LOC"] * (len(word) - 1))
            elif pos == 't':
                tag = "B-TIME"
                bio_tags.extend([tag] + ["I-TIME"] * (len(word) - 1))
            else:
                bio_tags.extend(["O"] * len(word))

    return bio_tags

def remove_special_tags(sentence):
    sentence=sentence.strip("\n")
    cleaned_sentence = re.sub(r'\/[a-z]+', '', sentence)
        # 去除句子中间的空格
    cleaned_sentence = cleaned_sentence.replace(' ', '')
    return cleaned_sentence

# def getbio(file):
#     data=read_raw(file)
#     data=[traditional_to_simplified(x)for x in data]
#     #用于进行实体标记的句子
#     sentence=[remove_special_tags(x)for x in data]
#     true=[convert_to_bio(x)for x in data]
#     return sentence,true

# train,reltrain=getbio(hantrainfile)
# dev,reldev=getbio(handevfile)
# test,y_true=getbio(hantestfile)

# print(test[:30])
#限于gpt4输出的结果并非是标准结果，还需要进一步处理。这里先用30条数据进行测试
# predictfile="data/result/nerresult/hantest.txt"
def readpredcit(file):
    enitylist=[]
    with open(file,encoding="utf-8") as f:
        lines=f.readlines()
        for line in lines:
            lst = ast.literal_eval(line)
            enitylist.append(lst)
    return enitylist

# predictenity=readpredcit(predictfile)
# for a,b,c in zip(test[:30],bioreltest[:30],predictenity):
#     print(a)
#     print(b)
#     print(c)

# 暂时只提供四种ofi标记
def bio_tagging(sentence,ners,type):
    type_tag={'per':['B-PER','I-PER'],"time":["B-TIME","I-TIME"],"loc":["B-LOC","I-LOC"],"ofi":["B-OFI","I-OFI"]}
    tag=type_tag[type]
    # 去除空格
    sentence = sentence.replace(' ', '')
    sentence = sentence.replace('」', '')
    sentence = sentence.replace('「', '')
    # 构建实体标注结果
    tagged_sentence = ['O'] * len(sentence)
    for entity in ners:
        start = 0
        while start < len(sentence):
            match = re.search(re.escape(entity), sentence[start:])
            if match:
                start_index = match.start() + start
                end_index = match.end() + start
                tagged_sentence[start_index] = tag[0]
                for i in range(start_index + 1, end_index):
                    tagged_sentence[i] = tag[1]
                start = end_index
            else:
                break
    return tagged_sentence

def bio_per_tagging(sentence, entities):
    # 去除空格
    sentence = sentence.replace(' ', '')
    # 构建实体标注结果
    tagged_sentence = ['O'] * len(sentence)
    for entity in entities:
        start = 0
        while start < len(sentence):
            match = re.search(re.escape(entity), sentence[start:])
            if match:
                start_index = match.start() + start
                end_index = match.end() + start
                tagged_sentence[start_index] = 'B-PER'
                for i in range(start_index + 1, end_index):
                    tagged_sentence[i] = 'I-PER'
                start = end_index
            else:
                break
    return tagged_sentence


def bio_loc_tagging(sentence, entities):
    # 去除空格
    sentence = sentence.replace(' ', '')

    # 构建实体标注结果
    tagged_sentence = ['O'] * len(sentence)
    for entity in entities:
        start = 0
        while start < len(sentence):
            match = re.search(re.escape(entity), sentence[start:])
            if match:
                start_index = match.start() + start
                end_index = match.end() + start
                tagged_sentence[start_index] = 'B-LOC'
                for i in range(start_index + 1, end_index):
                    tagged_sentence[i] = 'I-LOC'
                start = end_index
            else:
                break

    return tagged_sentence

def bio_time_tagging(sentence, entities):
    # 去除空格
    sentence = sentence.replace(' ', '')

    # 构建实体标注结果
    tagged_sentence = ['O'] * len(sentence)
    for entity in entities:
        start = 0
        while start < len(sentence):
            match = re.search(re.escape(entity), sentence[start:])
            if match:
                start_index = match.start() + start
                end_index = match.end() + start
                tagged_sentence[start_index] = 'B-TIME'
                for i in range(start_index + 1, end_index):
                    tagged_sentence[i] = 'I-TIME'
                start = end_index
            else:
                break

    return tagged_sentence


def get_per_enity(pereniy):
    pattern = r"\((.*?)\)"  # 匹配括号内的内容
    #输入实体
    matches = re.findall(pattern, pereniy)
    extracted_content = [match for match in matches]
    # print(extracted_content)
    # 用来存一个句子中答案人物实体
    temp = []
    # 通过()进行提取的人物，然后去掉人物 和标点符号剩下的字提出来
    for per in extracted_content:
        # per 为str，出现了未按照格式输出的句子
        if "人物" not in per:
            if "，" in per:
                e1, e2 = per.split("，")
                temp.append(e1)
                temp.append(e2)
            else:
                e1, e2 = per.split(",")
                temp.append(e1)
                temp.append(e2)
        else:
            pattern2 = r"\W"  # 匹配非字母数字字符
            perenity = re.sub(pattern2, "", per.replace("人物", ""))
            temp.append(perenity)
    return temp


def get_loc_enity(pereniy):
    pattern = r"\((.*?)\)"  # 匹配括号内的内容
    #输入实体
    matches = re.findall(pattern, pereniy)
    extracted_content = [match for match in matches]
    # 用来存一个句子中答案人物实体
    loclist = []
    # 通过()进行提取的人物，然后去掉人物 和标点符号剩下的字提出来
    for loc in extracted_content:
        # per 为str，出现了未按照格式输出的句子
        if "地点" not in loc:
            if "，" in loc:
                e1, e2 = loc.split("，")
                loclist.append(e1)
                loclist.append(e2)
            else:
                e1, e2 = loc.split(",")
                loclist.append(e1)
                loclist.append(e2)
        else:
            pattern2 = r"\W"  # 匹配非字母数字字符
            locenity = re.sub(pattern2, "", loc.replace("地点", ""))
            loclist.append(locenity)
    return loclist


def get_time_enity(pereniy):
    pattern = r"\((.*?)\)"  # 匹配括号内的内容
    #输入实体
    matches = re.findall(pattern, pereniy)
    extracted_content = [match for match in matches]
    # 用来存一个句子中答案人物实体
    timelist = []
    # 通过()进行提取的人物，然后去掉人物 和标点符号剩下的字提出来
    # def judge(enity):
    #     if enity =="无":
    #         return None
    #     else:
    #         return enity
    for time in extracted_content:
        # per 为str，出现了未按照格式输出的句子
        if "时间" not in time:
            if "，" in time:
                e1, e2 = time.split("，")
                #判断其中是不是无
                timelist.append(e1)
                timelist.append(e2)
            else:
                e1, e2 = time.split(",")
                timelist.append(e1)
                timelist.append(e2)
        else:
            pattern2 = r"\W"  # 匹配非字母数字字符
            timeenity = re.sub(pattern2, "", time.replace("时间", ""))
            timelist.append(timeenity)
    return timelist


def merge_bio_tags(per, loc,time,ofi):
    merged_tags = []
    for tag1, tag2 ,tag3,tag4 in zip(per,loc,time,ofi):
        if tag1 != 'O':
            merged_tags.append(tag1)
        elif tag2 != 'O':
            merged_tags.append(tag2)
        elif tag3!='O':
            merged_tags.append(tag3)
        else:
            merged_tags.append(tag4)
    return merged_tags

def merge_bio_tags2(*args):
    merged_tags = []
    for tags in zip(*args):  # 使用*args解包所有参数，并通过zip打包每一列
        for tag in tags:
            if tag != 'O':
                merged_tags.append(tag)
                break  # 找到非'O'标签后跳出内层循环
        else:  # 如果内层循环正常结束（即没有找到非'O'标签），执行下面的语句
            merged_tags.append('O')  # 添加'O'标签到结果中
    return merged_tags

def get_perenity_menzi(pereniy):
    if '(' in pereniy:
        if ')' not in pereniy:
            pereniy=pereniy+")"
    else:
        pass
    #有的时候，输出没有出现（）
    if "(" not in pereniy and ')' not in pereniy:
        pereniy="("+pereniy[4:]+')'
    else:
        pass
    temp = []
    tag=0
    searchstr = ["实体:无", "实体:空", "时间", "地理","无"]
    for str in searchstr:
        if str in pereniy:
            # 一旦出现直接break
            tag = 1
            break
        else:
            tag = 2

    if tag == 2:
        # 从括号内提取实体
        pattern = r"\((.*?)\)"  # 匹配括号内的内容
        matches = re.findall(pattern, pereniy)
        # print(matches)
        if matches[0]=='':
            pass
        else:
            extracted_content = [match for match in matches]
            # t5模型的输出符合格式的，extracted_content只有一位
            #多个实体，通过，号隔开
            if "," in extracted_content[0]:
                nerdata=extracted_content[0].split(",")
                nerdata = [x for x in nerdata if x != '']
                temp=temp+nerdata
            elif "，" in extracted_content[0]:
                nerdata = extracted_content[0].split("，")
                nerdata = [x for x in nerdata if x != '']
                temp = temp + nerdata
            else:
                temp=temp+extracted_content
    else:
        pass
    # print(pereniy)
    # print(temp)
    return temp

def get_locenity_menzi(locnerity):
    if '(' in locnerity:
        if ')' not in locnerity:
            locnerity=locnerity+")"
    else:
        pass
    temp = []
    tag=0
    searchstr = ["实体:无", "实体:空", "时间", "人物"]
    for str in searchstr:
        if str in locnerity:
            #一旦出现直接break
            tag=1
            break
        else:
            tag=2

    if tag==2:
        # 从括号内提取实体
        pattern = r"\((.*?)\)"  # 匹配括号内的内容
        matches = re.findall(pattern, locnerity)
        # print(matches)
        if matches[0]=='':
            pass
        else:
            extracted_content = [match for match in matches]
            # t5模型的输出符合格式的，extracted_content只有一位
            #多个实体，通过，号隔开
            if "," in extracted_content[0]:
                nerdata=extracted_content[0].split(",")
                nerdata = [x for x in nerdata if x != '']
                temp=temp+nerdata
            elif "，" in extracted_content[0]:
                nerdata = extracted_content[0].split("，")
                nerdata = [x for x in nerdata if x != '']
                temp = temp + nerdata
            else:
                temp=temp+extracted_content

    #tag不等于2的时候模型输出是错误的，直接返回空就行了
    else:
        pass
    # print(pereniy)
    # print(temp)
    return temp


def get_time_enity_menzi(timeenity):
    if '(' in timeenity:
        if ')' not in timeenity:
            timeenity=timeenity+")"
    else:
        pass
    temp = []
    tag=0
    searchstr=["实体:无","实体:空","地理","人物","地点","无"]
    #判断预测的实体中是不是出现了不想要的答案
    for str in searchstr:
        if str in timeenity:
            #一旦出现直接break
            tag=1
            break
        else:
            tag=2
    if tag==2:
        # 从括号内提取实体
        pattern = r"\((.*?)\)"  # 匹配括号内的内容
        matches = re.findall(pattern, timeenity)
        extracted_content = [match for match in matches]
        # t5模型的输出符合格式的，extracted_content只有一位
        #多个实体，通过，号隔开
        # print(extracted_content)
        if "," in extracted_content[0]:
            nerdata=extracted_content[0].split(",")
            nerdata = [x for x in nerdata if x != '']
            temp=temp+nerdata
        elif "，" in extracted_content[0]:
            nerdata = extracted_content[0].split("，")
            nerdata = [x for x in nerdata if x != '']
            temp = temp + nerdata
        #只有一个实体的时候就无需分割
        else:
            temp=temp+extracted_content
    #对应出现了没有实体，或者回答错误的情况
    else:
        pass
    # print(timeenity)
    # print(temp)
    return temp

def get_ofi_enity_guner(ofi):
    # 0 补全括号
    if '(' in ofi:
        if ')' not in ofi:
            ofi=ofi+")"
    else:
        pass

    # 1 看看有没有 无，或者其他标签
    temp = []
    tag=0
    searchstr=["实体:无","实体:空","地理","人物","地点","无"]
    #判断预测的实体中是不是出现了不想要的答案
    for str in searchstr:
        if str in ofi:
            #一旦出现直接break
            tag=1
            break
        else:
            tag=2

    if tag==2:
        # 从括号内提取实体
        pattern = r"\((.*?)\)"  # 匹配括号内的内容
        matches = re.findall(pattern, ofi)
        extracted_content = [match for match in matches]
        # t5模型的输出符合格式的，extracted_content只有一位
        #多个实体，通过，号隔开
        # print(extracted_content)
        if "," in extracted_content[0]:
            nerdata=extracted_content[0].split(",")
            nerdata = [x for x in nerdata if x != '']
            temp=temp+nerdata
        elif "，" in extracted_content[0]:
            nerdata = extracted_content[0].split("，")
            nerdata = [x for x in nerdata if x != '']
            temp = temp + nerdata
        #只有一个实体的时候就无需分割
        else:
            temp=temp+extracted_content

    #对应出现了没有实体，或者回答错误的情况，返回的就是空
    else:
        pass

    return temp

# 从bio文件读取 句子 和bio标注
def get_y_true(file):
    bio=readfile(file)
    y_true = []
    sentences = []
    for data in bio:
        # check len
        y_true.append(data["bio"])
        sentences.append(data["stentece"])
    print(len(y_true))
    print(len(sentences))
    return sentences,y_true

# 有时候会出现几条数据长短不一的情况，是由于回标和词汇替换产生的小问题。没几条，直接删了
def check_data(ss,bios,pres):
    slst=[]
    blst=[]
    prelst =[]
    i=0
    for s,bio,pre in zip(ss,bios,pres):
        lena = len(s)
        lenb = len(bio)
        lenc=len(pre)
        if lena==lenb and lenb==lenc:
            slst.append(s)
            blst.append(bio)
            prelst.append(pre)
        else:
            i+=1
    print(str(i)+"data wrong but not in ca")
    return slst,blst,prelst

# y_pred=[]
# for a,c in zip(test[:50],predictenity):
#     #先获得实体
#     # print(a)
#     per = get_per_enity(c[0])
#     loc = get_loc_enity(c[1])
#     time = get_time_enity(c[2])
#     #转换成BIo标签
#     perbio=bio_per_tagging(a,per)
#     # print(per)
#     # print(perbio)
#     locbio=bio_loc_tagging(a,loc)
#     # print(loc)
#     # print(locbio)
#     timebio=bio_time_tagging(a,time)
#     prediction=merge_bio_tags(perbio,locbio,timebio)
#     # print(prediction)
#     y_pred.append(prediction)
# y_true=y_true[:50]
# print(f1_score(y_true, y_pred))
# print(classification_report(y_true, y_pred))