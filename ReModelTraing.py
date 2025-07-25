from datasets import Dataset,DatasetDict
#加载模型分词
from Remodel import getmodel,get_tokenizer_and_basemodel
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import tensorflow as tf
import re
from Re_evaluate import getReport,savehistory
import numpy as np
from configs import dslabel,model2columns
#配置
batch_size=8
epochs=60
maxlen=256

def add_pos(examples):
    # 获取相对位置
    addp1=[]
    addp2=[]
    for example in examples['text']:
        sub_match = re.search(r'\[SUB_\w+\].*?\[/SUB_\w+\]', example)
        obj_match = re.search(r'\[OBJ_\w+\].*?\[/OBJ_\w+\]', example)
        # 获取 sub 实体的中心位置
        if sub_match:
            l1_start, l1_end = sub_match.span()
            l1_center = (l1_start + l1_end) // 2
        else:
            l1_center = 0  # 或者其他适当的默认值

        # 获取 obj 实体的中心位置
        if obj_match:
            l2_start, l2_end = obj_match.span()
            l2_center = (l2_start + l2_end) // 2
        else:
            l2_center = 0  # 或者其他适当的默认值
        p11 = []
        p22 = []
        for i, w in enumerate(example):
            a = i - l1_center
            b = i - l2_center
            if a > 50:
                a = 50
            if b > 50:
                b = 50
            if a < -50:
                a = -50
            if b < -50:
                b = -50
            p11.append(a +51)
            p22.append(b +51)
        a = maxlen - len(p11)
        if a > 0:
            front = int(a / 2)
            back = a - front
            front_vec = [0 for i in range(front)]
            back_vec = [0 for i in range(back)]
            p11 = front_vec + p11 + back_vec
            p22 = front_vec + p22 + back_vec
        else:
            p11 = p11[:maxlen]
            p22 = p22[:maxlen]
        addp1.append(p11)
        addp2.append(p22)
    return addp1,addp2


# 数据处理
def tokenize_and_align_labels(examples,tokenizer):
    tokenized_inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=maxlen)
    translation_tokenized = tokenizer(examples['translation'], truncation=True, padding='max_length', max_length=maxlen)
    examples['translation_input_ids'] = translation_tokenized['input_ids']
    examples['translation_token_type_ids'] = translation_tokenized['token_type_ids']
    examples['translation_attention_mask'] = translation_tokenized['attention_mask']
    pos1, pos2 = add_pos(examples)
    examples['pos1'] = pos1
    examples['pos2'] = pos2
    return tokenized_inputs

def get_tf_dataset(filepath,data_collator,tokenizer,columns):
    #从数据集读取数据
    ds=DatasetDict.load_from_disk(filepath)
    print(ds)
    # 使用 lambda 函数传递 tokenizer
    trainds = ds['train'].map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    valds = ds['validation'].map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    testds = ds['test'].map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    tf_train_dataset = trainds.to_tf_dataset(
        columns=columns,
        label_cols=["label"],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=batch_size,
        drop_remainder=True,
    )
    tf_val_dataset = valds.to_tf_dataset(
        columns=columns,
        label_cols=["label"],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=batch_size,
        drop_remainder=True,
    )
    tf_test_dataset=testds.to_tf_dataset(
        columns=columns,
        label_cols=["label"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=batch_size,
        drop_remainder=True,
    )
    return tf_train_dataset,tf_val_dataset,tf_test_dataset

def model_train(dspath,config):
    # 获取llm 分词器 和data_collator
    tokenizer,llm,data_collator=get_tokenizer_and_basemodel(config['lm'])

    #根据模型获取要加入的特征
    #获取数据集
    columns=model2columns[config['clsmodel']]
    tf_train_dataset,tf_val_dataset,tf_test_dataset=get_tf_dataset(dspath,data_collator,tokenizer,columns)

    #获取数据集的label数量，最终label数目加+2，后面试试
    fcsize=dslabel[config['ds']]
    clsmodel=getmodel(config['clsmodel'],llm,dp=0.5,fcsize=fcsize+2)
    # clsmodel=getmodel(config['clsmodel'],llm,dp=0.4,fcsize=fcsize+2)
    clsmodel.summary()
    # 1：complie
    lossfun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 改变optimizer
    num_train_step = len(tf_train_dataset) * epochs
    lr_scheduleer = PolynomialDecay(initial_learning_rate=3e-5, end_learning_rate=0.0, decay_steps=num_train_step)
    optimizer = tf.keras.optimizers.Adam(lr_scheduleer)

    """
    使用tensorborad
    """
    keras_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),

        # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', min_delta=0.01,restore_best_weights=True),

        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", verbose=2, patience=10, mode="max",restore_best_weights=True)
    ]

    clsmodel.compile(
        optimizer=optimizer,
        loss=lossfun,
        metrics=["accuracy"]
        # metrics=METRICS,
    )

    print("model - training ")
    history = clsmodel.fit(
        x=tf_train_dataset,
        epochs=epochs,
        verbose=2,
        callbacks=keras_callbacks,
        validation_data=tf_val_dataset,
    )

    basecheckpath = config["savemodelname"]
    # print(basecheckpath)
    # clsmodel.save(basecheckpath)
    # tf.saved_model.save()
    clsmodel.save_weights(basecheckpath)
    print("base Saved")

    """keras  compile fit predict"""
    # 获取测试集预测标签
    predics = clsmodel.predict(tf_test_dataset)
    prelabel = tf.math.argmax(predics, 1).numpy()
    # print(prelabel)

    #获取测试集真实标签
    true_labels = []
    for input_data, label in tf_test_dataset:
        true_labels.append(label.numpy())
    true_labels = np.concatenate(true_labels, axis=0)

    getReport(true_labels,prelabel,save=config['resultsavepath'])
    savehistory(history,filename=config['histroy'])
    print("over")

if __name__ =='__main__':
    # dsname cc or mascot
    # dsname = 'cc'
    dsname = 'chis'
    dsfloder = 'dataset/processed/re/' + dsname + '/'
    # lms = ['bert-ancient-chinese', 'bertbase', 'guwenbert-base', 'guwenbert-large', 'sikubert', 'sikuroberta']
    lms = ['bert-ancient-chinese', 'bertbase', 'guwenbert-base',  'sikubert', 'sikuroberta']
    #应当存在以下模型：基线（什么都不加），加位置信息，加语义信息，加ema和什么都加
    # clsmodel=['base','base_pos','base_semantic','base_resnet_EMA','base_pos_semantic_ema','base_pos_semantic']
    #已知base
    # clsmodel=['base','base_pos_resnet','base_resnet_EMA']
    clsmodel=['base_pos_resnet']
    configs = []
    for lm in lms:
        for cls in clsmodel:
            #配置应包含 dsname。lm，clsmodel，modelsavedpath，savedresult，以及history
            ds=dsname
            savedmodel = 'savedmodel/re/' + dsname + '-' + lm+'-'+cls+'/model'
            savedresult = 'savedResult/re/' + dsname + '-' + lm + '-'+cls+'.txt'
            historypath= 'savedResult/re/' + dsname + '-' + lm + '-'+cls+'history.txt'
            config = {'ds':dsname,"lm": lm, 'clsmodel':cls,"savemodelname": savedmodel, 'resultsavepath': savedresult,'histroy':historypath}
            configs.append(config)
    """
    自定义模型保存全部保存的是模型权重而不是整个模型
    """
    for config in configs:
        print(config)
        model_train(dspath=dsfloder,config=config)

    #单独训练
    # print(configs[0])
    # model_train(dspath=dsfloder,config=configs[0])



