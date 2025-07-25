from transformers import BertTokenizer,TFAutoModel,DataCollatorWithPadding,TFBertModel
import tensorflow as tf
from EMA import EMA
from configs import llmpath
import math
maxlen=256


METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]


# clsmodel=['base','basel_pos','base_semantic','base_ema','base_pos_semantic_ema']

def getmodel(modelname,llm,dp,fcsize):
    if modelname=='base':
        return basemodel(llm,dp,fcsize)
    elif modelname=='base_pos':
        return basemodelwithpos(llm,dp,fcsize)
    elif modelname=='base_pos_resnet':
        return base_pos_resnet(llm,dp,fcsize)
    elif modelname=='base_resnet_EMA':
        return base_resnet_EMA(llm,dp,fcsize)
    elif modelname=='base_pos_semantic':
        return base_pos_semantic(llm,dp,fcsize)
    else:
        raise ValueError("input go ⑧ ？")

def get_tokenizer_and_basemodel(checkpoint):
    checkpointpath=llmpath[checkpoint]
    tokenizer=BertTokenizer.from_pretrained(checkpointpath)
    bertmodel=TFAutoModel.from_pretrained(pretrained_model_name_or_path=checkpointpath,from_pt=True)
    print(bertmodel.config)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=maxlen, padding='max_length')
    return tokenizer,bertmodel,data_collator

#基础模型，什么都不加，直接语言模型输出
def basemodel(llm,dropout,fcsize):
    input_ids = tf.keras.layers.Input(shape=(256,), dtype=tf.int64, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(256,), dtype=tf.int64, name="attention_mask")
    outputs = llm(input_ids=input_ids, attention_mask=attention_mask,training=True)['pooler_output']
    outputs = tf.keras.layers.Dropout(dropout)(outputs)
    cla_outputs = tf.keras.layers.Dense(fcsize, activation=None)(outputs)
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=[cla_outputs])
    return model


def basemodelwithpos(llm, dropout,fcsize):
    input_ids = tf.keras.layers.Input(shape=(256,),batch_size=8, dtype=tf.int64, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(256,), batch_size=8,dtype=tf.int32, name="attention_mask")
    pos1 = tf.keras.layers.Input(shape=(256,),batch_size=8, dtype=tf.int32, name="pos1")  # 实体位置输入
    pos2 = tf.keras.layers.Input(shape=(256,),batch_size=8, dtype=tf.int32, name="pos2")  # 实体位置输入
    pos_embedding_layer=tf.keras.layers.Embedding(input_dim=102,output_dim=16,
                                                   embeddings_initializer=tf.keras.initializers.RandomUniform(
                                                       minval=-math.sqrt(6 / (3 * 32 + 3 * 768)),
                                                       maxval=math.sqrt(6 / (3 * 32 + 3 * 768))))
    # 假设 input_p1 和 input_p2 是你的输入张量
    input_x_p1 = pos_embedding_layer(pos1)
    input_x_p2 = pos_embedding_layer(pos2)

    # print(outputs.shape)#batchsize seqlen 768
    outputs = llm(input_ids=input_ids, attention_mask=attention_mask, training=True)['last_hidden_state']
    outputs = tf.keras.layers.Dropout(dropout)(outputs)
    x = tf.concat([outputs, input_x_p1, input_x_p2], 2)

    max_pool = tf.reduce_max(x, axis=1)
    avg_pool = tf.reduce_mean(x, axis=1)

    # attention_layer = tf.keras.layers.Attention()([outputs, input_x_p1_transformed, input_x_p2_transformed])
    # x = tf.keras.layers.LSTM(256, return_sequences=False)(x)
    pool =tf.concat([max_pool, avg_pool], axis=1)

    #加一层fc
    dense = tf.keras.layers.Dense(256, activation='relu')(pool)
    dense = tf.keras.layers.Dropout(dropout)(dense)
    # 分类器层
    cla_outputs = tf.keras.layers.Dense(fcsize, activation=None)(dense)
    model = tf.keras.Model(
        inputs=[input_ids, attention_mask, pos1, pos2],
        outputs=[cla_outputs]
    )
    return model


def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x


def base_pos_resnet(llm,dropout, fcsize):
    #加了符号标签的句子
    input_ids = tf.keras.layers.Input(shape=(256,), batch_size=8,dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(256,), batch_size=8,dtype=tf.int32, name="attention_mask")
    #翻译的句子
    # trans_input_ids = tf.keras.layers.Input(shape=(256,),batch_size=16, dtype=tf.int32, name="translation_input_ids")
    # trans_attention_mask = tf.keras.layers.Input(shape=(256,),batch_size=16, dtype=tf.int32, name="translation_attention_mask")

    #位置向量
    pos1 = tf.keras.layers.Input(shape=(256,), batch_size=8, dtype=tf.int32, name="pos1")  # 实体位置输入
    pos2 = tf.keras.layers.Input(shape=(256,), batch_size=8, dtype=tf.int32, name="pos2")  # 实体位置输入
    pos_embedding_layer = tf.keras.layers.Embedding(input_dim=102, output_dim=32,
                                                    embeddings_initializer=tf.keras.initializers.RandomUniform(
                                                        minval=-math.sqrt(6 / (3 * 32 + 3 * 768)),
                                                        maxval=math.sqrt(6 / (3 * 32 + 3 * 768))))
    # 假设 input_p1 和 input_p2 是你的输入张量
    input_x_p1 = pos_embedding_layer(pos1)
    input_x_p2 = pos_embedding_layer(pos2)

    # print(outputs.shape)#batchsize seqlen 768
    textbedding = llm(input_ids=input_ids, attention_mask=attention_mask, training=True)['last_hidden_state']
    textpool=llm(input_ids=input_ids, attention_mask=attention_mask, training=True)['pooler_output']
    textbedding = tf.keras.layers.Dropout(dropout)(textbedding)
    #词嵌入向量一定和pos连在一起用
    text_pos = tf.concat([textbedding, input_x_p1, input_x_p2], 2)
    #(b,256.768+32x2)
    # print(text_pos.shape)
    text_pos = tf.expand_dims(text_pos, -1) #-1表示最后一维

    #使用先使用一个卷积操作拓展维度
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(text_pos)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # (16, 128, 416, 64)
    print(x.shape)
    num_blocks = 2
    filters = 64
    for i in range(num_blocks):
        stride = 1
        if i % 4 == 0 and i != 0:
            filters *= 2
            stride = 2
        x = residual_block(x, filters, stride=stride)
    print('look')
    print(x.shape)
    print('look')
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # print('look')
    # print(x.shape)
    # print('look')
    x = tf.keras.layers.Dropout(dropout)(x)

    allfeature = tf.concat([x, textpool], axis=1)
    # print(allfeature.shape)
    cla_outputs = tf.keras.layers.Dense(fcsize, activation=None)(allfeature)
    model = tf.keras.Model(
        # inputs=[input_ids, attention_mask, pos1, pos2,trans_input_ids,trans_attention_mask],
        inputs=[input_ids, attention_mask, pos1, pos2,],
        outputs=[cla_outputs]
    )
    return model


def residual_block_EMA(x, filters, kernel_size=3, stride=1):
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    print('short shape')
    print(shortcut.shape)
    print('--------=')
    # 引入ema
    xema=EMA(shortcut.shape[3], factor=8)(shortcut)
    print('ema')
    print(xema.shape)
    x = tf.keras.layers.Add()([x, shortcut,xema])
    print('xadd')
    print(x.shape)
    x = tf.keras.layers.ReLU()(x)
    return x

def base_resnet_EMA(llm,dropout,fcsize):
    # 加了符号标签的句子
    input_ids = tf.keras.layers.Input(shape=(256,), batch_size=8, dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(256,), batch_size=8, dtype=tf.int32, name="attention_mask")

    # 位置向量
    pos1 = tf.keras.layers.Input(shape=(256,), batch_size=8, dtype=tf.int32, name="pos1")  # 实体位置输入
    pos2 = tf.keras.layers.Input(shape=(256,), batch_size=8, dtype=tf.int32, name="pos2")  # 实体位置输入
    pos_embedding_layer = tf.keras.layers.Embedding(input_dim=102, output_dim=32,
                                                    embeddings_initializer=tf.keras.initializers.RandomUniform(
                                                        minval=-math.sqrt(6 / (3 * 32 + 3 * 768)),
                                                        maxval=math.sqrt(6 / (3 * 32 + 3 * 768))))
    # 假设 input_p1 和 input_p2 是你的输入张量
    input_x_p1 = pos_embedding_layer(pos1)
    input_x_p2 = pos_embedding_layer(pos2)

    textbedding = llm(input_ids=input_ids, attention_mask=attention_mask, training=True)['last_hidden_state']
    textpool = llm(input_ids=input_ids, attention_mask=attention_mask, training=True)['pooler_output']
    textbedding = tf.keras.layers.Dropout(dropout)(textbedding)

    # 词嵌入向量一定和pos连在一起用
    text_pos = tf.concat([textbedding, input_x_p1, input_x_p2], 2)
    # (b,256.768+32x2)
    # print(text_pos.shape)
    text_pos = tf.expand_dims(text_pos, -1)  # -1表示最后一维

    # 使用先使用一个卷积操作拓展维度
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(text_pos)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # (16, 128, 416, 64)
    # print(x.shape)
    num_blocks = 2
    filters = 128
    for i in range(num_blocks):
        stride = 1
        if i % 4 == 0 and i != 0:
            filters *= 2
            stride = 2
        # (8, 128, 416, 64)
        x = residual_block(x, filters, stride=stride)
    #(8, 128, 416, 128)
    xema = EMA(x.shape[3], factor=8)(x)

    #换个方式引入ema
    xema=tf.keras.layers.GlobalAveragePooling2D()(xema)
    xema=tf.keras.layers.Dropout(dropout)(xema)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    # print('x.shape')
    # print(x.shape)
    # print('transpool')
    # print(transpool.shape)
    allfeatures = tf.concat([textpool,x,xema], axis=1)
    # print(allfeatures.shape)
    fc1=tf.keras.layers.Dense(256,activation="relu")(allfeatures)
    dp1 = tf.keras.layers.Dropout(dropout)(fc1)
    cla_outputs = tf.keras.layers.Dense(fcsize, activation=None)(dp1)

    model = tf.keras.Model(
        inputs=[input_ids, attention_mask, pos1, pos2],
        outputs=[cla_outputs])
    return model


def base_pos_semantic(llm,dropout,fcsize):
    # 加了符号标签的句子
    input_ids = tf.keras.layers.Input(shape=(256,), batch_size=8, dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(256,), batch_size=8, dtype=tf.int32, name="attention_mask")
    # 翻译的句子
    trans_input_ids = tf.keras.layers.Input(shape=(256,),batch_size=8, dtype=tf.int32, name="translation_input_ids")
    trans_attention_mask = tf.keras.layers.Input(shape=(256,),batch_size=8, dtype=tf.int32, name="translation_attention_mask")

    # # 位置向量
    # pos1 = tf.keras.layers.Input(shape=(256,), batch_size=8, dtype=tf.int32, name="pos1")  # 实体位置输入
    # pos2 = tf.keras.layers.Input(shape=(256,), batch_size=8, dtype=tf.int32, name="pos2")  # 实体位置输入
    # pos_embedding_layer = tf.keras.layers.Embedding(input_dim=102, output_dim=32,
    #                                                 embeddings_initializer=tf.keras.initializers.RandomUniform(
    #                                                     minval=-math.sqrt(6 / (3 * 32 + 3 * 768)),
    #                                                     maxval=math.sqrt(6 / (3 * 32 + 3 * 768))))
    # # 假设 input_p1 和 input_p2 是你的输入张量
    # input_x_p1 = pos_embedding_layer(pos1)
    # input_x_p2 = pos_embedding_layer(pos2)

    # textbedding = llm(input_ids=input_ids, attention_mask=attention_mask, training=True)['last_hidden_state']
    textpool = llm(input_ids=input_ids, attention_mask=attention_mask, training=True)['pooler_output']

    textbedding = tf.keras.layers.Dropout(dropout)(textpool)

    # transbedding=llm(input_ids=input_ids, attention_mask=attention_mask, training=True)['last_hidden_state']
    transpool=llm(input_ids=trans_input_ids, attention_mask=trans_attention_mask, training=True)['pooler_output']
    transbedding=tf.keras.layers.Dropout(dropout)(transpool)
    print(textbedding.shape)
    print(transbedding.shape)
    # 词嵌入向量一定和pos连在一起用
    textall = tf.concat([textbedding, transbedding], 1)
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(textall)


    fc1=tf.keras.layers.Dense(512,activation="relu")(x)
    fc1 = tf.keras.layers.Dropout(dropout)(fc1)

    fc2= tf.keras.layers.Dense(128, activation="relu")(fc1)
    fc2 = tf.keras.layers.Dropout(dropout)(fc2)

    cla_outputs = tf.keras.layers.Dense(fcsize, activation=None)(fc2)

    model = tf.keras.Model(
        inputs=[input_ids, attention_mask,trans_input_ids,trans_attention_mask],
        outputs=[cla_outputs])
    return model