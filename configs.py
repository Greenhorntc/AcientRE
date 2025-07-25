"用于存放一些关系标签等"
#moscot实体对应的
mascotlabel={"operate object":['per','thing'],
             "give birth to":['per','per'],
             "think of":['per','thing or per'],
             "contain":['loc','thing'],
             'come from':['thing or per','thing'],
             'inside':['loc','loc'],
             'attend':['thing','thing'],
             'buried in':['per','loc'],
             'leave':['per','thing or loc'],
             'attribute':['thing or per','thing'],
             'family member of':['per','per'],
             'take office':['per','ofi'],
             'speak':['per','thing'],
             'attack':['per','thing'],
             'defy':['thing','thing'],
             'appoint':['per','per'],
             'manage':['per','thing'],
             'recommend':['per','per'],
             'meet':['per','thing'],
             'request':['per','per'],
             'talk to':['per','per'],
             'worship':['per','per'],
             'appraise':['per','per'],
             'accept':['per','thing'],
             'create': ['thing', 'thing'],
             'tribute from':['thing','thing'],
             'stay at':['per','loc'],
             'alias':['per','per'],
             'married':['per','per'],
             'pardon':['per','per'],
             'give to':['per','thing or per'],
             'enfeoff':['per','thing or per'],
             }

mascotlabel2id={
            "operate object":0,
             "give birth to":1,
             "think of":2,
             "contain":3,
             'come from':4,
             'inside':5,
             'attend':6,
             'buried in':7,
             'leave':8,
             'attribute':9,
             'family member of':10,
             'take office':11,
             'speak':12,
             'attack':13,
             'defy':14,
             'appoint':15,
             'manage':16,
             'recommend':17,
             'meet':18,
             'request':19,
             'talk to':20,
             'worship':21,
             'appraise':22,
             'accept':23,
             'create': 24,
             'tribute from':25,
             'stay at':26,
             'alias':27,
             'married':28,
             'pardon':29,
             'give to':30,
             'enfeoff':31,
}

cclabel2id={
    '去往':0,
    '属于':1,
    '葬于':2,
    '朋友':3,
    '作':4,
    '字':5,
    '讨伐':6,
    '名':7,
    '号':8,
    '归属':9,
    '同名于':10,
    '依附':11,
    '管理':12,
    '姓':13,
    '父':14,
    '任职':15,
    '作战':16,
    '弟':17,
    '杀':18,
    '子':19,
    '隶属于':20,
    '位于':21,
    '出生地':22,
    '兄':23,
    '升迁':24
}

chislabel2id={
    '不存在关系':0,
    "敌对攻伐": 1,
    "任职": 2,
    "上下级": 3,
    "政治奥援": 4,
    "同僚": 5,
    "到达": 6,
    "管理": 7,
    "出生于某地": 8,
    "驻守": 9,
    "别名": 10,
    "父母": 11,
    "兄弟": 12,
}


llmpath={
    'bert-ancient-chinese': r"E:\Language_model\bert-ancient-chinese",
    'bertbase': r"E:\Language_model\bertbase",
    'guwenbert-base': r"E:\Language_model\guwenbert-base",
    'guwenbert-large': r"E:\Language_model\guwenbert-large",
    'sikubert': r"C:\Users\28205\PycharmProjects\LLMs\models\siku-bert",
    'sikuroberta': r"C:\Users\28205\PycharmProjects\LLMs\models\sikuroberta",
         }

dslabel={'cc':25,'mascot':32,'chis':13}


# ["attention_mask", "input_ids", 'pos1', 'pos2', 'translation_input_ids', 'translation_attention_mask']

# clsmodel=['base','basel_pos','base_semantic','base_ema','base_pos_semantic_ema']
model2columns={'base':["input_ids",'attention_mask'],
                'base_pos':["input_ids",'attention_mask', 'pos1', 'pos2'],
                'base_pos_resnet':["attention_mask", "input_ids", 'pos1', 'pos2'],
                'base_semantic':["input_ids",'attention_mask', 'translation_input_ids', 'translation_attention_mask'],
                'base_ema':["input_ids",'attention_mask'],
                'base_resnet_EMA':["attention_mask", "input_ids", 'pos1', 'pos2'],
                'base_pos_semantic':["attention_mask", "input_ids",  'translation_input_ids', 'translation_attention_mask']
               }

















