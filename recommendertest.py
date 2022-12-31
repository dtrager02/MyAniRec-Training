import torch
import pandas as pd
from time import perf_counter
# config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
#     model_file='EASE-Dec-22-2022_23-51-54.pth',
# )
model = torch.load("models/best_multvae.pt[100, 300, 10175]",map_location=torch.device('cpu'))
model.eval()
start = perf_counter()
isekai_animes = [29803,35790,30831]
comedy_animes = [37105,37999,36296]
shounen_anime = [1735,269,22199]
# converter = pd.read_csv("item_map.csv")
# model = torch.load("models/best_multvae.pt[100, 300, 10175]")
input = torch.zeros(3,10175)
converter = pd.read_csv("item_map.csv").set_index("item_id")
for i,l in enumerate([isekai_animes,comedy_animes,shounen_anime]):
    for j in l:
        j = converter.loc[j].values[0]
        input[i,j] = 1
out,_,_ = model(input)
out_sorted = torch.argsort(out,dim=1,descending=True)
out_sorted_converted = out_sorted.apply_(lambda x: converter[converter['train_id'] == x].index.values[0])
print(out_sorted_converted[:,0:10].tolist())
print(perf_counter()-start)
#current stats: |valid loss 33673.82| ndcg@100 0.1437 | hit rate@20 0.2220 | hit rate@50 0.3697|