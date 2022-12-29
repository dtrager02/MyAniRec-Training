from recbole.quick_start.quick_start import load_data_and_model
from  recbole.data.interaction import Interaction
import pandas as pd
import torch
# config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
#     model_file='EASE-Dec-22-2022_23-51-54.pth',
# )
model = torch.load("ease.pt",map_location=torch.device('cpu'))
isekai_animes = [29803,35790,30831]
comedy_animes = [37105,37999,36296]
shounen_anime = [1735,269,22199]
records = [[31234,1,1,8000]]
# converter = pd.read_csv("item_map.csv").set_index("item_id")
# for i,l in enumerate([isekai_animes]):
#     records.extend([[i,converter.loc[s]['train_id'],1,8000] for s in l])
records = pd.DataFrame(records,columns=['user_id', 'item_id', 'rating', 'timestamp']).reset_index(drop=True)
print(records)
records2 = Interaction(records)
print(records2)
print(model)
out = model.full_sort_predict(records2)
print(out.size(),torch.argsort(out,dim=1,descending=True))
# for param in model.parameters():
#     print(param,param.size())
