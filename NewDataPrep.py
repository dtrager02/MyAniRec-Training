import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
class GeneralDataLoader(Dataset):
    def __init__(self,data,batch_size=4096,n_items=None,mode="LOO",side_info= None):
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.mode = mode
        self.n_users = 0
        if torch.cuda.is_available():  
          self.device = torch.device("cuda:0")
        if n_items is not None:
            self.n_items = n_items
        else:
            self.n_items = data.loc[:,'anime_id'].max()
        #create code to reindex the username column
        mapper = dict(zip(data['username'].unique(),[*range(data['username'].nunique())]))
        data.loc[:,'username'] = data.loc[:,'username'].map(mapper)
        self.side_info = side_info
        if side_info is not None: 
            self.side_info.loc[:,'username'] = self.side_info.loc[:,'username'].map(mapper)
            self.side_info.set_index("username",inplace=True)
            self.side_info = self.side_info/self.side_info.max() #normalize
        del mapper
        self.n_users = data.loc[:,'username'].max()+1
        
        #set negative feedback to -1 instead of 0
        data.loc[data['train_id'] == 0,'train_id'] = -1
        
        self.sparse_ratings2,self.sparse_ratings3 = self.get_dataset(data)
        self.row_order = np.arange(self.n_users)
        np.random.shuffle(self.row_order)
        self.row_order = torch.tensor(self.row_order,device=self.device,dtype=torch.int64)
        # self.row_order = torch.tensor(self.row_order,device=self.device)
        
    def get_side_info(self,indices):
        arr = self.side_info.loc[indices.cpu(),:].values
        return torch.tensor(arr,device=self.device,dtype=torch.float32)
    
    def __len__(self):
        return self.n_users//self.batch_size-1 #-1 is for index out of bounds errors

    def __getitem__(self, idx):
        raise NotImplementedError
    def get_dataset(self,data):
        raise NotImplementedError
    

class TestDataloader(GeneralDataLoader):
 
    def __init__(self, data, batch_size, n_items=None,mode = "LOO",side_info=None):
        super().__init__(data, batch_size,n_items=n_items,mode=mode,side_info=side_info)
        
    def __getitem__(self, idx):
        #shuffle if train and on last batch
        if idx == self.__len__():
            raise StopIteration
        indices = self.row_order[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        if self.side_info is not None:
            side_info = self.get_side_info(indices)
        else:
            side_info = None
        a_ = torch.index_select(self.sparse_ratings2,0,indices) #self.sparse_ratings2[indices]
        a = a_.to_dense()
        b_ = torch.index_select(self.sparse_ratings3,0,indices)
        b = b_.to_dense()
        return a,b,side_info
    
    def get_dataset(self,data:pd.DataFrame):   
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
        
        sparse_ratings2 = torch.sparse_coo_tensor(torch.tensor(data.loc[:,["username","anime_id"]].values.T),\
            torch.tensor(data.loc[:,["train_id"]].values.flatten()),size=(self.n_users,self.n_items),dtype=torch.float32,device="cpu").coalesce().to(self.device)
        sparse_ratings3 = torch.sparse_coo_tensor(torch.tensor(last_ones_df.loc[:,["username","anime_id"]].values.T),\
            torch.tensor(last_ones_df.loc[:,["train_id"]].values.flatten()),size=(self.n_users,self.n_items),dtype=torch.float32,device="cpu").coalesce().to(self.device)
        return sparse_ratings2,sparse_ratings3
    

class TrainDataloader(GeneralDataLoader):


    def __init__(self, data, batch_size, n_items=None,side_info=None):
        super().__init__(data, batch_size,n_items=n_items,mode=None,side_info=side_info)
  
    def __getitem__(self, idx):
        #shuffle if train and on last batch
        if idx == self.__len__():
            # tqdm.write("shuffling users in train")
            self.row_order = np.arange(self.n_users)
            np.random.shuffle(self.row_order)
            self.row_order = torch.tensor(self.row_order,device=self.device,dtype=torch.int64)
            raise StopIteration
        indices = self.row_order[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        if self.side_info is not None:
            side_info = self.get_side_info(indices)
        else:
            side_info = None
        a_ = torch.index_select(self.sparse_ratings2,0,indices)
        a = a_.to_dense()
        return a,side_info
    

    
    def get_dataset(self,data:pd.DataFrame):   
        # last_ones = last_ones_df.to_numpy()
        # data = data.to_numpy()
        #create a torch sparse tensor with rows as usernames and columns as item ids
        sparse_ratings2 = torch.sparse_coo_tensor(torch.tensor(data.loc[:,["username","anime_id"]].values.T),\
            torch.tensor(data.loc[:,["train_id"]].values.flatten()),size=(self.n_users,self.n_items),dtype=torch.float32,device="cpu").coalesce().to(self.device)
        print("Sparse tensor created, shape = ",sparse_ratings2.size())
        return sparse_ratings2,None