import torch
import pandas as pd
from time import perf_counter,time
import asyncio
import random
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

def timefunc(input,model,converter):
    # global model
    # global converter
    start = time()
    out,_,_ = model(input)
    out_sorted = torch.argsort(out,dim=1,descending=True)
    out_sorted_converted = out_sorted.apply_(lambda x: converter[converter['train_id'] == x].index.values[0])
    t = time()-start
    return out_sorted_converted[:,0:10].tolist(),t

def initializer(model1,converter1):
    global model
    global converter
    model = model1
    converter = converter1

async def producer(input,executor,loop):
    # background_tasks = set()
    # print("in")
    # for i in range(100):
    #     # task = asyncio.create_task(asyncio.wrap_future(, loop=None
    #     task = loop.run_in_executor(executor, timefunc, input)
    #     # print("task made")
    #     # Add task to the set. This creates a strong reference.
    #     background_tasks.add(task)

    #     # To prevent keeping references to finished tasks forever,
    #     # make each task remove its own reference from the set after
    #     # completion:
    #     task.add_done_callback(consumer1)
    a = 0
    futures = [executor.submit(timefunc,input) for i in range(20)]
    for s in concurrent.futures.as_completed(futures):
        a += s.result()[1]
    return a/20.0
        
# def consumer1(f):
#     global outputlist
#     outputlist.append(f.result()[1])
#     print("done",str(f.result()[1]))

async def main(loop,executor,input):
    global outputlist
    a = await producer(input,executor,loop)
    print(a)
    # a = await asyncio.sleep(5)
    # print(sum(outputlist)/len(outputlist))

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
    converter2 = pd.read_csv("item_map.csv")
    for i,l in enumerate([isekai_animes,comedy_animes,shounen_anime]):
        for j in l:
            j = converter.loc[j].values[0]
            input[i,j] = 1
    input = torch.vstack([input[:,:] for i in range(2)])
    print(input.size())
    start = perf_counter()
    # for i in range(10):
    out,_,_ = model(input)
    out_sorted = torch.argsort(out,dim=1,descending=True)
    print(perf_counter()-start)
    out_sorted_converted2 = converter.reset_index().set_index("train_id").loc[out_sorted[0,:10],"item_id"].values
    print(perf_counter()-start)
    out_sorted_converted = out_sorted.apply_(lambda x: converter[converter['train_id'] == x].index.values[0])[0,:10].tolist()
    print(perf_counter()-start)
    print(out_sorted_converted)
    print(out_sorted_converted2)
    assert out_sorted_converted == out_sorted_converted2
    
    # outputlist = []
    # executor = ProcessPoolExecutor(initializer=initializer,initargs=(model,converter))
    # loop = asyncio.get_event_loop()
    # outputlist = []
    # loop.run_until_complete(main(loop,executor,input))
    # loop.close()