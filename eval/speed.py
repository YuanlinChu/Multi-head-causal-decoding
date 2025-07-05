import json
from transformers import AutoTokenizer
import numpy as np

# tokenizer=AutoTokenizer.from_pretrained("/hpc2hdd/home/ychu763/Documents/my_model/shareGPT_mhc_vicuna-7b-v1.3_headsnum_5_lr_0.001_layersnum_1")
tokenizer=AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
jsonl_file = "data/mt_bench/model_answer/vicuna-mhc-7b-v1.3-shareGPT-temperature-0.0-posterior_threshold-0.09-posterior_alpha-0.3.jsonl"
jsonl_file_base = "data/mt_bench/model_answer/vicuna-mhc-7b-v1.3-greedy.jsonl"
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)



speeds=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens



# print('speed',np.array(speeds).mean())
# print('speed0',np.array(speeds0).mean())
print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())