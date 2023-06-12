import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from NewDataPrep import *
from metric import *
from model2 import MultiVAE,loss_function
from torch.utils.data import DataLoader
import wandb
import random
import yaml
from collections import defaultdict
import traceback
import os 
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"


config1 = yaml.load(open("config.yaml","r"),Loader=yaml.FullLoader)
print(config1)
sweep_id = wandb.sweep(config1, project="recsys")
device = torch.device("cpu")
if torch.cuda.is_available():  
   device = torch.device("cuda:0")
print(device)
###############################################################################
# Weights and Biases
###############################################################################
def train_wrapper(config=None):
   run = wandb.init(config=config)
   try:
      train()
   except Exception as e:
      print(traceback.format_exc())


def train(config=None):
   config = wandb.config
   lr = config.lr
   batch_size = config.batch_size
   epochs = config.epochs
   metric_names = ["ndcg@100","bi_ndcg@100" ,"hit_rate@20","bi_hit_rate@20"]
   ###############################################################################
   # Load data
   ###############################################################################
   data = pd.read_feather("ratings_ipc_processed.ipc")
   side_info = False
   # side_info = pd.read_feather("user_stats.ipc")
   users = data['username'].unique()
   np.random.shuffle(users)
   splits = np.split(users,[int(.8*len(users)),int(.9*len(users))])
   train_users,val_users,test_users = data[data['username'].isin(splits[0])],data[data['username'].isin(splits[1])],data[data['username'].isin(splits[2])]
   # train_side,val_side,test_side = side_info[side_info['username'].isin(splits[0])],side_info[side_info['username'].isin(splits[1])],side_info[side_info['username'].isin(splits[2])]
   print("split data")
   
   #n_items safer to obtain from item_map.csv
   train_loader = TrainDataloader(train_users,batch_size=batch_size,n_items=10662)
   # test_loader =TestDataloader(test_users,batch_size=batch_size,mode="80/20",n_items=10661)
   valid_loader = TestDataloader(val_users,batch_size=batch_size//2,mode="80/20",n_items=10662) #,side_info=val_side.copy(deep=True)
   n_items = train_loader.n_items
   # idxlist = list(range(N))
   print("Data loaded")
   
   ###############################################################################
   # Build the model
   ###############################################################################

   anneal_cap = config.anneal_cap

   q_dims = [n_items+ (side_info.shape[1]-1) if side_info else 0,config.layer1, config.layer2]
   p_dims = [config.layer2, config.layer1, n_items]
   total_anneal_steps = 10000
   del data,users,train_users,val_users,test_users
   #del train_side,val_side,test_side
   model = MultiVAE(p_dims,dropout=config.dropout).to(device)
   print(f"Model: {p_dims}, dropout: {config.dropout}")
   optimizer = optim.Adam(model.parameters(), lr=lr)
   criterion = loss_function

   ###############################################################################
   # Training code
   ###############################################################################

   
   # Turn on training mode
   model.train()
   scaler = torch.cuda.amp.GradScaler()
   total_anneal_steps = (len(train_loader) * epochs) // (1.5*anneal_cap)
   update_count = 0
   best_n100 = -np.inf
   train_loss_hist = torch.zeros(epochs)
   # np.random.shuffle(idxlist)
   positives,negative = torch.zeros((batch_size,n_items),device=device),torch.zeros((batch_size,n_items),device=device)
   for epoch in range(1, epochs + 1):
      train_loss = 0.0
      for data,side in iter(tqdm(train_loader,position=0, leave=True)):
         positives.zero_()
         negative.zero_()
         positives[data==1] = 1
         negative[data==-1] = 1
         if side is not None:
            data = torch.cat([data,side],1)
         # print(data.shape,data[0].shape)
         if total_anneal_steps > 0:
            anneal = min(anneal_cap, 
                             1. * update_count / total_anneal_steps)
         else:
            anneal = anneal_cap 
         optimizer.zero_grad()
         
         with torch.cuda.amp.autocast(dtype=torch.float16):
            recon_batch, mu, logvar = model(data)
            loss = criterion(recon_batch, positives, mu, logvar, anneal)
         scaler.scale(loss).backward()
         train_loss += loss.item()
         scaler.step(optimizer)
         scaler.update() 
         update_count += 1
      wandb.log({"train total_loss":train_loss})
      tqdm.write(f"Epoch {epoch}: train loss {train_loss}, anneal: {anneal}")
      metrics = evaluate(model, valid_loader,anneal,metric_names,prefix="validation ")
      # Save the model if the n100 is the best we've seen so far.
      if metrics["validation bi_ndcg@100"] > best_n100:
         with open("models/best_multvae.pt"+str(model.p_dims), 'wb') as f:
            torch.save(model.state_dict(), f)
            tqdm.write("Saving model (new best validation n100)")
            #wand file upload
            
         best_n100 = metrics["validation bi_ndcg@100"]
      
      #Check for increases in ndcg for early stopping
      # train_loss_hist[epoch - 1] = -val_ndcg
      # diffs = torch.diff(train_loss_hist)
      # if (diffs > 0).sum() > 4:
      #    tqdm.write("Early stopping")
      #    return
      # evaluate(model, test_loader,anneal,metric_names,prefix="test ")
   art = wandb.Artifact(f"multvae-{wandb.run.id}", type="model")
   art.add_file("models/best_multvae.pt"+str(model.p_dims))
   wandb.log_artifact(art)
   wandb.finish()


def evaluate(model, valid_loader,anneal,metric_names,criterion=loss_function,prefix="test "):
   # Turn on evaluation mode
   model.eval()
   metric_logs = defaultdict(list)
   total_loss = 0.0
   positives,negative = torch.zeros((valid_loader.batch_size,valid_loader.n_items),device=device),torch.zeros((valid_loader.batch_size,valid_loader.n_items),device=device)
   positives_heldout,negatives_heldout = torch.zeros((valid_loader.batch_size,valid_loader.n_items),device=device),torch.zeros((valid_loader.batch_size,valid_loader.n_items),device=device)
   with torch.no_grad():
         for data,heldout_data,side in iter(tqdm(valid_loader,position=0, leave=True,mininterval=2.0,miniters=1)):
            positives.zero_()
            negative.zero_()
            positives_heldout.zero_()
            negatives_heldout.zero_()
            positives[data==1] = 1
            negative[data==-1] = 1
            positives_heldout[heldout_data==1] = 1
            negatives_heldout[heldout_data==-1] = 1
            if side is not None:
               data = torch.cat([data,side],1)
            
            recon_batch, mu, logvar = model(data)
            loss = criterion(recon_batch, positives, mu, logvar, anneal)
            total_loss += loss.item()
            # Exclude examples from training set because they will not be recommended
            recon_batch[positives == 1] = -torch.inf
            metrics = Metric(recon_batch,positives_heldout,negatives_heldout,metric_names,device=device)
            res = metrics.calculate()
            for k,v in res.items():
               metric_logs[k].append(v)
   metric_logs_out = dict()
   for k,v in metric_logs.items():
      metric_logs_out[prefix+k] = np.mean(v)
   metric_logs_out[prefix+'total_loss'] = total_loss
   wandb.log(metric_logs_out)
   tqdm.write(Metric.prettify(metric_logs_out))
   model.train()
   return metric_logs_out

wandb.agent(sweep_id, train_wrapper, count=16)