import torch
class Metric:
    def __init__(self, pred:torch.Tensor, actual:torch.Tensor,bad_actual:torch.Tensor=None,metrics: list = ['hit_rate@10','mrr','ndcg@10'],device = 'cpu'):
        self.pred = pred
        self.device = device
        self.actual = actual
        self.bad_actual = bad_actual
        self.metrics = metrics
        #Valid metrics: Hit rate, MRR, Recall, NDCG, 
        self.sorted_pred = torch.argsort(pred, dim=1, descending=True)
        
    @staticmethod
    def first_nonzero(a):
        #get the index of the first nonzero element of each row in a tensor a   
        mask = (a != 0).to(torch.int32)
        return mask.argmax(1).unsqueeze(1)
        
    def hit_rate(self,k,extra:torch.Tensor=None):
        # print(self.actual[self.sorted_pred[:,:k].unsqueeze(2)])
        # print(torch.sum(self.actual[:,self.sorted_pred[:,:k]], dim=1))
        if extra is not None:
            arr = extra
        else:
            arr = self.actual
        return torch.count_nonzero(torch.sum(arr.gather(1,self.sorted_pred[:,:k]), dim=1))/arr.size()[0]
    
    def bi_hit_rate(self,k):
        #positives
        pos = self.hit_rate(k)
        #negatives
        neg = self.hit_rate(k,self.bad_actual)
        return torch.mean(pos-neg)
    def mrr(self):
        a = self.actual.gather(1,self.sorted_pred)
        nz = self.first_nonzero(a)+1 #+1 because the index starts from 0
        return torch.sum(torch.ones_like(nz)/nz)/nz.size()[0]
    def ndcg(self,k,extra:torch.Tensor=None):
        if extra is not None:
            arr = extra
        else:
            arr = self.actual
        temp = arr.gather(1,self.sorted_pred[:,:k])
        dcg = torch.sum(temp/torch.log2(torch.arange(2,k+2,device=self.device)), dim=1)
        temp2 = torch.sort(arr, dim=1, descending=True).values[:,:k]
        temp2[:,0] = 1 #ensures there is at least 1 positive
        idcg = torch.sum(temp2/torch.log2(torch.arange(2,k+2,device=self.device)), dim=1)
        return torch.mean(dcg/idcg)
    def bi_ndcg(self,k):
        #positives
        pos = self.ndcg(k)
        #negatives
        neg = self.ndcg(k,self.bad_actual)
        return torch.mean(pos-neg)
    def calculate(self):
        res = {}
        for metric in self.metrics:
            try:
                metric, k = metric.split('@')
                k = int(k)
                if metric == 'hit_rate':
                    res[metric+'@'+str(k)] = self.hit_rate(k).item()
                elif metric == 'bi_hit_rate':
                    res[metric+'@'+str(k)] = self.bi_hit_rate(k).item()
                elif metric == 'recall':
                    res[metric+'@'+str(k)] = self.recall(k).item()
                elif metric == 'ndcg':
                    res[metric+'@'+str(k)] = self.ndcg(k).item()
                elif metric == 'bi_ndcg':
                    res[metric+'@'+str(k)] = self.bi_ndcg(k).item()
            except ValueError: #catch those without @k
                if metric == 'mrr':
                    res[metric] = self.mrr().item()
            
        return res
    @staticmethod
    def prettify(res):
        out = " | ".join([f"{k}:{v:.4f}" for k,v in res.items()])
        return out
    
if __name__ == "__main__":
    actual = torch.tensor([[1, 0, 1,0,0], [0, 0, 1,0,1]])
    bad = torch.zeros_like(actual)
    bad[0,-2] = 1
    pred = torch.tensor([[0.1, 0.2, 0.3,.7,.1], [0.2, 0.1, 0.3,.5,.6]])
    metrics = Metric(pred, actual, bad,['hit_rate@3', 'mrr','ndcg@1','bi_hit_rate@3','bi_ndcg@1'])
    print(metrics.calculate())

    