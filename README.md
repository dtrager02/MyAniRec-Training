# MyAniRec-Training

This repository contains all the code required to train a recommender system efficiently using MultVAE, as well as loading and processing anime data. Some parts will be generalized later to work with any dataset. The datasets used are not in this respositoy. The model architecture and some parts of the training algorithm were copied from [younggyoseo/vae-cf-pytorch](https://github.com/younggyoseo/vae-cf-pytorch).

## File Descriptions
-  NewDataPrep.py: a efficient subclass of torch.utils.data.DataSet. It stores ratings as a torch sparse float tensor and slices the tensor to deliver data to the model. It also renumbers and shuffles users each time the data is iterated through. Can be used with torch.utils.data.DataLoader efficiently.
- metric.py: Efficient calculator for hit rate@k, mrr, and ndcg@k. Currently missing recall because it is equivalent to hit rate @k using the leave-one-out sampling strategy. 
- model.py: Contains MultVAE model architecture
- recommendertest.py and runmodel.py: Quick and dirty test for some imaginary users. recommendertest.py is outdates as  I no longer use RecBole due to performance issues.
- setup.ipynb: Data preprocessing
- train.py: Self explanatory
