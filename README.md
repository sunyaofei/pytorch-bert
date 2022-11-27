# pytorch
This is a repository for a few projects built in torch.

# submodule
```sh
# 大文件存储支持
yum install git-lfs

# https://huggingface.co/bert-base-uncased/tree/main => use in transformers
# https://huggingface.co/bert-base-uncased
git submodule add --name thirdparty/bert-base-uncased https://huggingface.co/bert-base-uncased
cd bert-base-uncased
git lfs pull
```

# 中文数据集参考
> https://blog.csdn.net/alip39/article/details/95891321
```
9.simplifyweibo_4_moods数据集：
36 万多条，带情感标注 新浪微博，包含 4 种情感，其中喜悦约 20 万条，愤怒、厌恶、低落各约 5 万条

使用方式详见：
https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/simplifyweibo_4_moods/intro.ipynb
```
![](./weibo情况语料)