import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class GeneralDataLoader(Dataset):
    def __init__(self,data,batch_size=4096,n_items=None,mode="LOO"):
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.mode = mode
        if torch.cuda.is_available():  
          self.device = torch.device("cuda:0")
          torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if n_items is not None:
            self.n_items = n_items
        self.sparse_ratings2,self.sparse_ratings3 = self.get_dataset(data)
        self.row_order = np.arange(self.n_users)
        np.random.shuffle(self.row_order)
        self.row_order = torch.tensor(self.row_order,device=self.device,dtype=torch.int64)
        # self.row_order = torch.tensor(self.row_order,device=self.device)
    def __len__(self):
        return self.n_users//self.batch_size
    def __getitem__(self, idx):
        raise NotImplementedError
    def get_dataset(self,data):
        raise NotImplementedError
    

class TestDataloader(GeneralDataLoader):
 
    def __getitem__(self, idx):
        #shuffle if train and on last batch
        indices = self.row_order[idx:idx+self.batch_size]
        a = torch.index_select(self.sparse_ratings2,0,indices)
        a = a.to_dense()
        b = torch.index_select(self.sparse_ratings3,0,indices)
        b = b.to_dense()
        return a,b
    
    def get_dataset(self,data:pd.DataFrame):   
        #create code to reindex the username column and make a two sparse matrices implimenting the leave one out split strategy 
        self.n_users = data.loc[:,'username'].nunique()

        #get the last rating for each user
        if self.mode == "LOO":
            last_ones = data.groupby("username",sort=False)["timestamp"].transform(pd.Series.max)
            last_ones_df = data[data['timestamp']==last_ones]
            data = data[data['timestamp']!=last_ones]
        elif self.mode == "80/20":
            last_ones = data.groupby("username",sort=False)["timestamp"].transform(pd.Series.quantile,0.8)
            last_ones_df = data[data['timestamp']>=last_ones]
            data = data[data['timestamp']<last_ones]
        # last_ones = last_ones_df.to_numpy()
        # data = data.to_numpy()
        #create a torch sparse tensor with rows as usernames and columns as item ids

        if self.n_items is None:
            self.n_items = data[:,1].max()+1
        sparse_ratings2 = torch.sparse_coo_tensor(torch.tensor(data.loc[:,["username","anime_id"]].values),\
            torch.tensor(data.loc[:,["score"),size=(self.n_users,self.n_items),dtype=torch.float32,device=self.device)
        sparse_ratings3 = torch.sparse_coo_tensor(torch.tensor(np.array([last_ones[:,0],last_ones[:,1]])),\
            torch.tensor(last_ones[:,2]),size=(self.n_users,self.n_items),dtype=torch.float32,device=self.device)
        return sparse_ratings2,sparse_ratings3
    

class TrainDataloader(Dataset):

    def __init__(self,data,batch_size=4096,n_items=None):
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        if torch.cuda.is_available():  
          self.device = torch.device("cuda:0")
          torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if n_items is not None:
            self.n_items = n_items
        self.sparse_ratings2 = self.get_dataset(data)
        self.row_order = np.arange(self.n_users)
        np.random.shuffle(self.row_order)
        self.row_order = torch.tensor(self.row_order,device=self.device,dtype=torch.int64)
        # self.row_order = torch.tensor(self.row_order,device=self.device)
    def __len__(self):
        return self.n_users//self.batch_size
  
    def __getitem__(self, idx):
        #shuffle if train and on last batch
        indices = self.row_order[idx:idx+self.batch_size]
        a = torch.index_select(self.sparse_ratings2,0,indices)
        a = a.to_dense()
        if idx == self.__len__()-1:
            print("shuffling users in train")
            self.row_order = np.arange(self.n_users)
            np.random.shuffle(self.row_order)
            self.row_order = torch.tensor(self.row_order,device=self.device,dtype=torch.int64)
        return a
    

    
    def get_dataset(self,data):   
        #create code to reindex the username column and make a two sparse matrices implimenting the leave one out split strategy 
        users = data.loc[:,'username'].unique()
        mapper = dict(zip(users,range(len(users))))
        self.n_users = len(users)
        data['username'] = data['username'].map(mapper)
        data[data['score'] > 0].loc[:,"score"] = 1
        #get the last rating for each user
        # last_ones = data.groupby("username",sort=False)["timestamp"].transform(pd.Series.max)
        # last_ones_df = data[data['timestamp']==last_ones]
        # data = data[data['timestamp']!=last_ones]
        # last_ones = last_ones_df.to_numpy()
        data = data.to_numpy()
        #create a torch sparse tensor with rows as usernames and columns as item ids
        #0 is username
        #1 is anime_id
        #2 is score
        if self.n_items is None:
            self.n_items = data[:,1].max()+1
        sparse_ratings2 = torch.sparse_coo_tensor(torch.tensor(np.array([data[:,0],data[:,1]])),\
            torch.tensor(data[:,2]),size=(len(users),self.n_items),dtype=torch.float32,device=self.device)

        return sparse_ratings2