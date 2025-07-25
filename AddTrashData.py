"""
执行前 务必 务必 务必 备份好temp/chis文件。

"""

import os
from ReDataHandler import Re_data_handler

datahandler=Re_data_handler()
# datahandler.CHisIEC_data_temp()
# datahandler.data_deal(dsname='chis')
# trainchanged=datahandler.change_CHisIEC_data(train)
# # print(trainchanged)

#read temp test

folder='dataset/processed/re/retemp/chis/'
files=os.listdir(folder)
files=[folder+x for x in files]
test, train, val=datahandler.read_data_temp(files)
print(test)
print(test[0])
print(len(test))
print(len(test)%8)
#因为测试的时候batchsize 设置为8 所以有的测试数据无法完全被预测。drop 增加几条废弃数据用于整除 一共加三条数据
text_tag='[CLS]该条数据为废弃数据，由<SUB_PER>摸鱼之王</SUB_PER>创建，<OBJ_PER>唐朝</OBJ_PER>用于满足数据总量被批大小整除。评估时请删除[SEP]'
txt_tras1='废弃数据，废弃数据，废弃数据，重要事情说三遍。'
label=10
txt_tras2='废弃数据，废弃数据，废弃数据，重要事情说三遍。'
trashdata=[text_tag,txt_tras1,label,txt_tras2]
test.append(trashdata)
test.append(trashdata)
print("ooo")
print(len(test)%8)
# testfile=r'D:\PythonPro\AncientIE\dataset\processed\re\retemp\chis\test_816.json'
testfile=r'D:\PythonPro\AncientIE\dataset\processed\re\retemp\chis\test.json'
datahandler.sava_data_to_json(test,testfile)

datahandler.data_deal(dsname='chis')
