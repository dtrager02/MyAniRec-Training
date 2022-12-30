import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from NewDataPrep import *
from metric import *
from model import MultiVAE,loss_function
from torch.utils.data import DataLoader
###############################################################################
# Load data
###############################################################################
device = torch.device("cpu")

if torch.cuda.is_available():  
   device = torch.device("cuda:0")
print(device)

train = pd.read_feather("data/train_processed.feather")
test = pd.read_feather("data/test_processed.feather")
valid  = pd.read_feather("data/valid_processed.feather")
train_loader = TrainDataloader(train,batch_size=4096,n_items=10175)
test_loader = DataLoader(TestDataloader(test,batch_size=4096,n_items=10175))
valid_loader = DataLoader(TestDataloader(valid,batch_size=4096,n_items=10175))
n_items = train_loader.n_items
train_loader = DataLoader(train_loader)
# idxlist = list(range(N))
print("Data loaded")
###############################################################################
# Build the model
###############################################################################
best_n100 = -np.inf
anneal_cap = 0.2
log_interval = 2
update_count = 0
total_anneal_steps = 200000
p_dims = [128, 384, n_items]
model = MultiVAE(p_dims).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
criterion = loss_function

###############################################################################
# Training code
###############################################################################

def train(epochs,dataloader,optimizer,criterion):
   # Turn on training mode
   model.train()
   
   start_time = time.time()
   global update_count
   global best_n100
   # np.random.shuffle(idxlist)
   for epoch in range(1, epochs + 1):
      train_loss = 0.0
      for data in iter(tqdm(dataloader,position=0, leave=True)):
         if total_anneal_steps > 0:
             anneal = min(anneal_cap, 
                             1. * update_count / total_anneal_steps)
         else:
             anneal = anneal_cap 
         optimizer.zero_grad()
         recon_batch, mu, logvar = model(data[0])

         loss = criterion(recon_batch, data[0], mu, logvar, anneal)
         loss.backward()
         train_loss += loss.item()
         optimizer.step()  
         update_count += 1
      n100, r20, r50, loss = evaluate(model, valid_loader)
      tqdm.write('|epoch {:3d} |train loss {:4.2f} \n|valid loss {:4.2f}| ndcg@100 {:.4f} | hit rate@20 {:.4f} | hit rate@50 {:.4f}|'
                 .format(epoch,train_loss,loss,n100,r20,r50))

      # Save the model if the n100 is the best we've seen so far.
      if n100 > best_n100:
         with open("best_multvae.pt"+str(model.p_dims), 'wb') as f:
            torch.save(model, f)
         best_n100 = n100
   
   n100, r20, r50, test_loss = evaluate(model, test_loader)
   tqdm.write('|test loss {:4.2f} | ndcg@100 {:.4f} | hit rate@20 {:.4f} | hit rate@50 {:.4f}|'
                 .format(test_loss,n100,r20,r50))


def evaluate(model, valid_loader):
    # Turn on evaluation mode
    model.eval()
    n100_list = []
    r20_list = []
    r50_list = []
    total_loss = 0.0
    metrics = Metric
    with torch.no_grad():
         for data,heldout_data in iter(tqdm(valid_loader,position=0, leave=True,mininterval=2.0,miniters=1)):
            if total_anneal_steps > 0:
                anneal = min(anneal_cap, 
                               1. * update_count / total_anneal_steps)
            else:
                anneal = anneal_cap

            recon_batch, mu, logvar = model(data[0])

            loss = criterion(recon_batch, data[0], mu, logvar, anneal)
            total_loss += loss.item()
            # Exclude examples from training set
            # recon_batch = recon_batch.cpu().numpy()
            b = data[0].nonzero()
            recon_batch[b[:,0],b[:,1]] = -torch.inf
            
            metrics = Metric(recon_batch, heldout_data[0], ["ndcg@100", "hit_rate@20", "hit_rate@50"])
            res = metrics.calculate()

            n100_list.append(res['ndcg@100'])
            r20_list.append(res['hit_rate@20'])
            r50_list.append(res['hit_rate@50'])
 
    n100_list = np.array(n100_list)
    r20_list = np.array(r20_list)
    r50_list = np.array(r50_list)

    return np.mean(n100_list), np.mean(r20_list), np.mean(r50_list),total_loss

def gridsearch():
   p_dims = [[i, i*3, n_items] for i in (100, 150, 200)]
   for s in p_dims:
      model = MultiVAE(s).to(device)
      optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
      criterion = loss_function
      train(25,train_loader,optimizer,criterion)
# except KeyboardInterrupt:
#     print('-' * 89)
#     print('Exiting from training early')

# # Load the best saved model.
# with open(args.save, 'rb') as f:
#     model = torch.load(f)

# # Run on test data.
# test_loss, n100, r20, r50 = evaluate(test_data_tr, test_data_te)
# print('=' * 89)
# print('| End of training | test loss {:4.2f} | n100 {:4.2f} | r20 {:4.2f} | '
#         'r50 {:4.2f}'.format(test_loss, n100, r20, r50))
# print('=' * 89)

gridsearch()