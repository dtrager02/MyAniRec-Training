import torch
class Metric:
    def __init__(self, pred:torch.tensor, actual:torch.tensor,metrics: list,k=[10,20]):
        self.pred = pred
        self.actual = actual
        self.metrics = metrics
        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.IntTensor)
        #Valid metrics: Hit rate, MRR, Recall, NDCG, 
        self.sorted_pred = torch.argsort(pred, dim=1, descending=True)
        self.k = k
        
    @staticmethod
    def first_nonzero(a):
        #get the index of the first nonzero element of each row in a tensor a   
        mask = (a != 0).to(torch.int32)
        return mask.argmax(1).unsqueeze(1)
        
    def hit_rate(self,k):
        # print(self.actual[self.sorted_pred[:,:k].unsqueeze(2)])
        # print(torch.sum(self.actual[:,self.sorted_pred[:,:k]], dim=1))
        return torch.count_nonzero(torch.sum(self.actual.gather(1,self.sorted_pred[:,:k]), dim=1))/self.actual.size()[0]
    def mrr(self):
        a = self.actual.gather(1,self.sorted_pred)
        nz = self.first_nonzero(a)+1 #+1 because the index starts from 0
        
        return torch.sum(torch.ones_like(nz)/nz)/nz.size()[0]
    def ndcg(self,k):
        temp = self.actual.gather(1,self.sorted_pred[:,:k])
        dcg = torch.sum(temp/torch.log2(torch.arange(2,k+2).unsqueeze(1)), dim=1)
        temp2 = torch.sort(self.actual, dim=1, descending=True).values[:,:k]
        idcg = torch.sum(temp2/torch.log2(torch.arange(2,k+2).unsqueeze(1)), dim=1)
        return torch.mean(dcg/idcg)
    def calculate(self):
        res = {}
        for metric in self.metrics:
            if metric == 'hit_rate':
                for k in self.k:
                    res[metric+'@'+str(k)] = self.hit_rate(k)
            elif metric == 'mrr':
                res[metric] = self.mrr()
            elif metric == 'recall':
                res[metric] = self.recall()
            elif metric == 'ndcg':
                for k in self.k:
                    res[metric+'@'+str(k)] = self.ndcg(k)
        return res
    
if __name__ == "__main__":
    actual = torch.tensor([[1, 0, 1,0,0], [0, 0, 1,0,0]])
    pred = torch.tensor([[0.1, 0.2, 0.3,.7,.1], [0.2, 0.1, 0.3,.5,.6]])
    metrics = Metric(pred, actual, ['hit_rate', 'mrr','ndcg'], k=[1,2])
    print(metrics.calculate())

    