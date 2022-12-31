import torch
import requests
import numpy as np
import pandas as pd
def fetch_user(session:requests.Session,user:str):
    headers = {
    'authority': 'myanimelist.net'
    }
    
    headers['Referer'] = f'https://myanimelist.net/animelist/{user.strip()}'
    a = []
    offset = 0
    r = session.get(f"https://myanimelist.net/animelist/{user.strip()}/load.json?offset={offset}&status=7",headers=headers)  
    try:
        res_json = r.json()
        if "errors" not in res_json: 
            a.extend(res_json)
            offset = len(a)
        while len(a)%300==0 and len(a) and len(res_json):
            # time.sleep(random.random()/10)
            r = session.get(f"https://myanimelist.net/animelist/{user.strip()}/load.json?offset={offset}&status=7",headers=headers)  
            res_json = r.json()
            a.extend(res_json)
            offset += len(a)
        return {'status':r.status_code,'content':a,'user':user}
    except:
        return {'status':r.status_code,'content':None,'user':user}
    
def generate_user_tensor(user:str,n_items,item_map):
    session = requests.Session()
    res = fetch_user(session,"EscanorPie")
    out = torch.zeros((1,n_items))
    res = np.array([[s['anime_id'],s['score']] for s in res['content'] if s['status'] == 2])
    res = res[res[:,1]>np.median(res[:,1]),0]
    res = item_map.loc[res,"train_id"].values
    out[0,res] = 1
    return out,len(res)
    
if __name__ == "__main__":
    session = requests.Session()
    item_map = pd.read_csv("item_map.csv").set_index("item_id")
    out = generate_user_tensor("EscanorPie",10175,item_map)
    print(out)