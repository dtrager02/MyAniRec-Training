import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class GeneralDataLoader(Dataset):
    def __init__(self,data,batch_size=4096,n_items=None,mode="LOO"):
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.mode = mode
        self.n_users = 0
        if torch.cuda.is_available():  
          self.device = torch.device("cuda:0")
        if n_items is not None:
            self.n_items = n_items
        else:
            self.n_items = None
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
 
    def __init__(self, data, batch_size, n_items=None,mode = "LOO"):
        super().__init__(data, batch_size,n_items=n_items,mode=mode)
        
    def __getitem__(self, idx):
        #shuffle if train and on last batch
        if idx == self.__len__():
            raise StopIteration
        indices = self.row_order[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        a = torch.index_select(self.sparse_ratings2,0,indices) #self.sparse_ratings2[indices]
        a = a.to_dense()
        b = torch.index_select(self.sparse_ratings3,0,indices)
        b = b.to_dense()
        return a,b
    
    def get_dataset(self,data:pd.DataFrame):   
        #create code to reindex the username column and make a two sparse matrices implimenting the leave one out split strategy 
        self.n_users = data.loc[:,'username'].max()+1
        data.loc[data['train_id'] == 0,'train_id'] = -1
        last_ones_df = pd.DataFrame()
        #get the last rating for each user
        if self.mode == "LOO":
            last_ones = data.groupby("username",sort=False)["created_at"].transform(pd.Series.max)
            last_ones_df = data[data['created_at']==last_ones]
            data = data[data['created_at']!=last_ones]
        elif self.mode == "80/20":
            last_ones = data.groupby("username",sort=False)["created_at"].transform(pd.Series.quantile,0.8)
            last_ones_df = data[data['created_at']>=last_ones]
            data = data[data['created_at']<last_ones]
        # last_ones = last_ones_df.to_numpy()
        # data = data.to_numpy()
        #create a torch sparse tensor with rows as usernames and columns as item ids

        if self.n_items is None:
            self.n_items = data.loc[:,'anime_id'].max()
        sparse_ratings2 = torch.sparse_coo_tensor(torch.tensor(data.loc[:,["username","anime_id"]].values.T),\
            torch.tensor(data.loc[:,["train_id"]].values.flatten()),size=(self.n_users,self.n_items),dtype=torch.float32,device="cpu").coalesce().to(self.device)
        sparse_ratings3 = torch.sparse_coo_tensor(torch.tensor(last_ones_df.loc[:,["username","anime_id"]].values.T),\
            torch.tensor(last_ones_df.loc[:,["train_id"]].values.flatten()),size=(self.n_users,self.n_items),dtype=torch.float32,device="cpu").coalesce().to(self.device)
        return sparse_ratings2,sparse_ratings3
    

class TrainDataloader(GeneralDataLoader):


    def __init__(self, data, batch_size, n_items=None):
        super().__init__(data, batch_size,n_items=n_items,mode=None)
  
    def __getitem__(self, idx):
        #shuffle if train and on last batch
        if idx == self.__len__():
            # print("shuffling users in train")
            self.row_order = np.arange(self.n_users)
            np.random.shuffle(self.row_order)
            self.row_order = torch.tensor(self.row_order,device=self.device,dtype=torch.int64)
            raise StopIteration
        indices = self.row_order[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        a = torch.index_select(self.sparse_ratings2,0,indices)
        a = a.to_dense()
        return a
    

    
    def get_dataset(self,data:pd.DataFrame):   
        #create code to reindex the username column and make a two sparse matrices implimenting the leave one out split strategy 
        self.n_users = data.loc[:,'username'].max()+1
        data.loc[data['train_id'] == 0,'train_id'] = -1
        # last_ones = last_ones_df.to_numpy()
        # data = data.to_numpy()
        #create a torch sparse tensor with rows as usernames and columns as item ids
        if self.n_items is None:
            self.n_items = data.loc[:,'anime_id'].max()
        sparse_ratings2 = torch.sparse_coo_tensor(torch.tensor(data.loc[:,["username","anime_id"]].values.T),\
            torch.tensor(data.loc[:,["train_id"]].values.flatten()),size=(self.n_users,self.n_items),dtype=torch.float32,device="cpu").coalesce().to(self.device)
        print("Sparse tensor created, shape = ",sparse_ratings2.size())
        return sparse_ratings2,None