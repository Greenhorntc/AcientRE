import tensorflow as tf
from sklearn.metrics import recall_score,precision_score,f1_score,classification_report

def getReport(rellabels,pre_label,save):

    matrix = tf.keras.metrics.Accuracy()
    matrix.update_state(rellabels, pre_label)
    acc = matrix.result().numpy()
    print("测试ACC")
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

    labels = list(range(0,13))
    # labels = list(range(0,25))
    print("report")
    print(classification_report(rellabels, pre_label, labels=labels,digits=5))
    with open(file=save, encoding="UTF-8", mode='w') as f:
        f.write(str(float(acc)) + ":测试ACC ")
        f.write("\n")
        f.write(str(float(Precision)) + ":测试Precision ")
        f.write("\n")
        f.write(str(float(recall)) + ":测试Recall")
        f.write("\n")
        f.write(str(float(f1)) + ":测试F1")
        f.write("\n")
        f.close()

def datatostr(data):
    datstr="@"
    for i in data:
        datstr=datstr+str(i)
    return datstr

def savehistory(history,filename):
    epoch=datatostr(history.epoch)
    loss=datatostr(history.history["loss"])
    val_loss= datatostr(history.history["val_loss"])
    acc=datatostr(history.history["accuracy"])
    valacc=datatostr( history.history["val_accuracy"])
    with open(file=filename,encoding="UTF-8",mode='w') as f:
        f.write(epoch+" ")
        f.write("\n")
        f.write(loss+" ")
        f.write("\n")
        f.write(val_loss+" ")
        f.write("\n")
        f.write(acc+" ")
        f.write("\n")
        f.write(valacc + " ")
        f.write("\n")
        f.close()