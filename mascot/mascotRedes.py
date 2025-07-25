'''
根据label 获取 redes
'''

from LLMRe.configs import mascotlabel2id

def get_re_des_file(labelfile):
    with open(labelfile,encoding='utf-8') as f:
        strall=f.read()
    return strall

labellst=list(mascotlabel2id.keys())
mascotdic={}
for label in labellst:
    labelfile='redes/{}.txt'.format(label)
    redes=get_re_des_file(labelfile)
    # print(redes)
    mascotdic[label]=redes

# print(mascotdic)