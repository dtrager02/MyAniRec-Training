import pandas as pd
import polars as pl
import timeit
import random
def create_map(path="ratings_ipc_processed.ipc"):
    df = pd.read_feather(path)
    print(df)
    uniques = df['anime_id'].unique()
    print(len(uniques))
    mapper = dict(zip(uniques,range(len(uniques))))

    c = pd.DataFrame(zip(uniques,[*range(len(uniques))]),columns=['item_id','train_id'])
    c.to_csv("item_map.csv")
    df['anime_id'] = df['anime_id'].map(mapper)
    df_polars = pl.from_pandas(df)
    df_polars.write_ipc("ratings_ipc_processed.ipc")

def check_id_density(path="ratings_ipc_processed.ipc"):
    df = pd.read_feather(path)
    uniques = df['anime_id'].unique()
    maximum = uniques.max()
    return len(uniques)/(maximum+1)
def compare_pandas_polars(path="item_map.csv"):
    df = pd.read_feather(path)
    polars_df = pl.read_ipc(path,memory_map=False)
    df.set_index('train_id',inplace=True)
    a = df.loc[[0,100,50,70],"item_id"]
    b = pl.DataFrame({"a":[0,100,50,70]}).join(polars_df,left_on="a",right_on="train_id",how="left").get_column("item_id").to_numpy()
    c = polars_df.filter(pl.col("train_id").is_in([0,100,50,70])).select("item_id")
    
class IdConverter:
    def __init__(self,path) -> None:
        self.df = pl.read_ipc(path,memory_map=False)
    
    def to_train_id(self,item_ids):
        return pl.DataFrame({"a":item_ids}).join(self.df,left_on="a",right_on="item_id",how="left").get_column("train_id").to_numpy()
    def to_item_id(self,train_ids):
        return pl.DataFrame({"a":train_ids}).join(self.df,left_on="a",right_on="train_id",how="left").get_column("item_id").to_numpy()

if __name__ == "__main__":
    create_map()
    print(check_id_density())
    # converter = IdConverter("item_map.csv")
    # a = converter.to_item_id([0,100,50,70])
    # expect [12149, 30749, 38680, 7054]
    # b = converter.to_train_id(a)
    # print(type(b),len(b),b)
    