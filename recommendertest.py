import torch
import pandas as pd
from time import perf_counter
import asyncio
import random
from concurrent.futures import ProcessPoolExecutor
# config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
#     model_file='EASE-Dec-22-2022_23-51-54.pth',
# )




# traced_model = torch.jit.load("RecBol/models/traced_model.pt",map_location=torch.device('cpu'))#
# with open("models/traced_model.pt", "wb") as f:
#     torch.jit.save(traced_model, f)
# input = torch.vstack([input[:,:] for i in range(50)])
# script_model = torch.jit.script(model)
# traced_model = torch.jit.trace(model, input)
# print(input.size())
# start = perf_counter()
# out2,_,_ = traced_model(input)
# print(perf_counter()-start)
def timefunc(input):
    global model
    global converter
    start = perf_counter()
    out,_,_ = model(input)
    out_sorted = torch.argsort(out,dim=1,descending=True)
    out_sorted_converted = out_sorted.apply_(lambda x: converter[converter['train_id'] == x].index.values[0])
    t = perf_counter()-start
    return out_sorted_converted[:,0:10].tolist(),t

def initializer(model1,converter1,inputQueue1,outputQueue1):
    global model
    global converter
    global inputQueue
    global outputQueue
    inputQueue = inputQueue1
    outputQueue = outputQueue1
    model = model1
    converter = converter1

async def producer(queue,input):
    for i in range(10):
        await queue.put(input)
        print("appended")
        await asyncio.sleep(random.random()/1000.0)
    return -1
        
async def consumer1(in_queue,executor,loop):
    count = 0
    times = 0
    while count < 9:
        input = await in_queue.get()
        out,t = await loop.run_in_executor(executor, timefunc, input)
        print("done",str(t))
        count +=1
        times+=t
    return times/count

# assert (out2 != out).sum()==0

# model = torch.load("RecBol/models/best_multvae.pt[100, 300, 10175].pt",map_location=torch.device('cuda:0'))
# model.eval()
# input = input.to("cuda:0")
# traced_model = torch.jit.trace(model, input)
# start = perf_counter()
# for _ in range(20):
#     out,_,_ = traced_model(input)
# print((perf_counter()-start)/20)
# print(out)
#current stats: |valid loss 33673.82| ndcg@100 0.1437 | hit rate@20 0.2220 | hit rate@50 0.3697|

async def main(loop,executor,input,queue):
    consumers = []
    for _ in range(16):
        consumer = asyncio.create_task(consumer1(queue))
        consumers.append(consumer)

    a = await asyncio.gather(producer(queue,input,executor,loop),*consumers, return_exceptions=True)
    print(a)

if __name__ == "__main__":
    model = torch.load("models/best_multvae.pt[100, 300, 10175].pt",map_location=torch.device('cpu'))
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
    input = torch.vstack([input[:,:] for i in range(2)])
    print(input.size())
    inputQueue = asyncio.Queue()
    outputQueue = asyncio.Queue()
    executor = ProcessPoolExecutor(initializer=initializer,initargs=(model,converter,inputQueue,outputQueue))
    loop = asyncio.get_event_loop()
    
    loop.run_until_complete(main(loop,executor,input,inputQueue))
    loop.close()