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