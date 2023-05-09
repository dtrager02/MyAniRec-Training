import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from NewDataPrep import *
import pandas as pd
from torch.utils.data import DataLoader

train = pd.read_feather("ratings_ipc_processed.ipc")
print(train['train_id'].value_counts())
# groups = train.groupby(["username","anime_id"]).count().loc[:,["train_id"]]
# print(groups.max(),groups.min())
# train.drop_duplicates(subset=["user","anime_id"],keep=False,inplace=True)
# train.reset_index().to_feather("ratings_ipc.ipc")
# print(train[train.duplicated(subset=["user","anime_id"],keep=False)].sort_values(by=["user","anime_id"]))
# print(train.loc[:,["train_id"]].values.flatten())
# print(train.loc[:,["username","anime_id"]].values.T.shape)
train_loader = TestDataloader(train,batch_size=5,mode="80/20",n_items=10661)

print(len(train_loader))
# a = next(iterator).to("cpu")#.type(torch.int64) + 1
# b = next(iterator).to("cpu")#.type(torch.int64) + 1
# print(a.min(),a.max())
# print(a.sum(dim=1),"\n",b.sum(dim=1))
count = 0
for s in train_loader:
    #train
    # print(s.size())
    # print(np.bincount(s[0].flatten().type(torch.int64).cpu().numpy()+1,minlength=3))
        #train
    print(s[0].size(),s[1].size())
    print(np.bincount(s[0][0].flatten().type(torch.int64).cpu().numpy()+1,minlength=3),np.bincount(s[1][0].flatten().type(torch.int64).cpu().numpy()+1,minlength=3))
