import os
d = os.path.dirname(__file__)
import sys
sys.path.append(d)

from transformers import BertTokenizer
from bert_sst2 import BertSST2Model
import bert_sst2 as bt



def preprocess(raw_data_path, data_path):
    formated = open(data_path, 'w')
    cnt=1
    lines = []
    with open(raw_data_path, 'r') as raw:
        for l in raw.readlines():
            if not l[0] in '0123':
                continue
            # formated.write(l[0] + '\t' + l[2:])
            lines.append(l[0] + '\t' + l[2:])
            cnt += 1
    import random
    random.shuffle(lines)
    formated.writelines(lines)
    formated.close()


def eval(pretrained_model_name, model_path, texts):
    # pretrained_model_name = pwd + '/bert-base-uncased'
    # tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    inputs = bt.tokenizer(texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512)
    
    moods = ['喜悦', '愤怒', '厌恶', '低落']
    
    import torch
    model=bt.BertSST2Model(len(moods), pretrained_model_name)
    m=torch.load(model_path)
    model.load_state_dict(m)
    model.eval()
    
    # model = torch.load(model_path)
    r = model(inputs)
    print(r)
    print(r.argmax(dim=1))

    indice = r.argmax(dim=1)
    return [moods[i] for i in indice]

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
# save/load 只是属性/整个模型
if __name__ == '__main__':
    pwd = os.getenv("PWD")
    raw_data_path = pwd + '/bert-sst2/simplifyweibo_4_moods.csv'
    data_path = pwd + '/bert-sst2/simplifyweibo_4_moods_formated.csv'
    pretrained_model_name = pwd + '/bert-base-uncased'
    bt.pretrained_model_name = pretrained_model_name
    bt.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    
    # 数据预处理
    # preprocess(raw_data_path, data_path)
    
    # train_dataloader, test_dataloader, categories = bt.load(data_path, num=5000)
    # bt.train(train_dataloader, test_dataloader, categories, num_epoch=1)
    
    # train_dataloader, test_dataloader, categories = bt.load(data_path, num=None)
    # bt.train(train_dataloader, test_dataloader, categories, num_epoch=5)
    
    model_path = '/root/nlp/pytorch-bert/bert_sst2_12_08_09_57/checkpoints-1/model_state.pth'
    # model_path = '/root/nlp/pytorch-bert/bert_sst2_11_27_23_17/checkpoints-1/model.pth'
    mood = eval(pretrained_model_name, model_path, 
                [
                    '不会水的旱鸭子在这里…咦,最右边那个看起来我不认识的...,肥星占了半张照片,林广还啃玉米!!!~ ~ ~ 好暗啊~ 都看不怎么清我们的俊容了~ ~ 凳凳凳灯...红花湖游泳照终于出炉啦~ ~ ~'
                 ])
    print(mood)
    
    

    
    