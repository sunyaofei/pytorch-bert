import os
d = os.path.dirname(__file__)
import sys
sys.path.append(d)

from transformers import BertTokenizer
from bert_sst2 import BertSST2Model


# https://pytorch.org/tutorials/beginner/saving_loading_models.html
# save/load 只是属性/整个模型
if __name__ == '__main__':
    pwd = os.getenv("PWD")
    pretrained_model_name = pwd + '/bert-base-uncased'
    # 加载预训练模型对应的tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    
    
    inputs = tokenizer(["ms . phoenix is completely lacking in charm and charisma , and is unable to project either esther's initial anomie or her eventual awakening ."],
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    
    import torch
    model=torch.load('/root/nlp/pytorch-bert/bert_sst2_11_23_12_31/checkpoints-5/model.pth')
    model.eval()
    r = model(inputs)
    print(r)
    # tensor([[-0.6965,  0.1081]], grad_fn=<AddmmBackward0>)
    print(r.argmax(dim=1))
    # tensor([1])
    print(r.argmax(dim=1).item())
    # 1
    
    
    # mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')

    # from huggingface_hub import snapshot_download
    # from transformers import AutoModel
    # snapshot_download(repo_id="bert-base-chinese", cache_dir="./tmp")
    # snapshot_download(repo_id="bert-base-chinese", cache_dir="./tmp", ignore_regex=['*.h5', '*.ot', '*.msgpack'])
    # AutoModel.from_pretrained("bert-base-chinese",  mirror='https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models')
    # AutoModel.from_pretrained("bert-base-chinese",  mirror='tuna')
    
    
    