import typer
import json
from typing_extensions import Annotated
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import os
from multiprocessing import Pool, Manager
import multiprocessing

app = typer.Typer()

class ModelWrapper:
    def __init__(self, model_name="Medusa/mistralai/Mistral-7B-Instruct-v0.2", device_ids=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 修改设备映射逻辑
        if device_ids is not None:
            # 将模型完整地加载到指定的单个 GPU 上
            self.device = f'cuda:{device_ids[0]}'
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map={"": self.device},  # 将整个模型放在一个 GPU 上
                local_files_only=True,
                use_cache=True,
                low_cpu_mem_usage=True,
            )
        else:
            self.device = "cuda:0"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map={"": self.device},
                local_files_only=True,
                use_cache=True,
                low_cpu_mem_usage=True,
            )
        
        self.model.eval()
        
        # 设置最大长度限制
        self.max_input_length = 4000
        self.max_total_tokens = 4096
        self.max_batch_prefill_tokens = 4000

    @torch.inference_mode()
    def batch_generate_response(self, batch_messages):
        """批量生成回复"""
        # 将每个对话格式化为 Mistral 格式
        prompts = []
        for messages in batch_messages:
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"[INST] {msg['content']} [/INST]"
                else:
                    prompt += msg['content']
            prompts.append(prompt)
        
        # 批量编码输入
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length
        ).to(self.model.device)
        
        # 批量生成回复
        outputs = self.model.generate(
            **inputs,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            top_k=0,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_total_tokens,
            min_length=1,
            use_cache=True,
            repetition_penalty=1.0,
            return_dict_in_generate=False,
            output_scores=False,
        )
        
        # 解码生成的回复
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 提取最后的回复
        results = []
        for response in responses:
            last_response = response.split("[/INST]")[-1].strip()
            results.append({"role": "assistant", "content": last_response})
            
        return results

def fix_source(source):
    if source and source[0]["from"] == "gpt":
        source = source[1:]
    new_source = []
    for item in source:
        role = "assistant" if item["from"] == "gpt" else "user"
        content = item["value"]
        new_source.append({"role": role, "content": content})
    return new_source

def process_conversation_batch(batch, model_wrapper, pbar):
    """真正的批量处理对话"""
    # 收集每个对话当前的状态
    current_states = [([], conv) for conv in batch]  # (历史消息, 剩余对话)
    conversation_histories = [[] for _ in batch]
    
    while current_states:
        # 准备这一批次的输入
        current_inputs = []
        active_indices = []
        for i, (history, remaining) in enumerate(current_states):
            user_messages = [msg for msg in remaining if msg["role"] == "user"]
            if not user_messages:
                continue
            
            current_messages = history + [user_messages[0]]
            current_inputs.append(current_messages)
            active_indices.append(i)
        
        if not current_inputs:
            break
        
        # 批量生成回复
        responses = model_wrapper.batch_generate_response(current_inputs)
        
        # 更新状态
        for idx, response in zip(active_indices, responses):
            history, remaining = current_states[idx]
            user_msg = [msg for msg in remaining if msg["role"] == "user"][0]
            new_history = history + [user_msg, response]
            conversation_histories[idx].extend([user_msg, response])
            
            new_remaining = [msg for msg in remaining[remaining.index(user_msg)+1:]]
            
            remain_user_msg = [msg for msg in new_remaining if msg["role"] == "user"]
            if remain_user_msg:
                current_states[idx] = (new_history, new_remaining)
            else:
                current_states[idx] = ([], [])
        
        # 移除已完成的对话
        current_states = [(h, r) for h, r in current_states if r]
    
    # 批次处理完成后更新进度条
    pbar.update(len(batch))
    
    return conversation_histories

from multiprocessing import Queue

def process_chunk(args):
    """处理一个数据块的函数"""
    chunk, gpu_id, model_name, batch_size, save_interval, output_path, start_idx, result_queue = args
    print(f"GPU {gpu_id} 开始处理 {len(chunk)} 条对话")
    
    model_wrapper = ModelWrapper(model_name=model_name, device_ids=[gpu_id])
    
    pbar = tqdm(
        total=len(chunk),
        desc=f"GPU {gpu_id}",
        position=gpu_id + 1,
        leave=True
    )
    
    # 处理对话并记录每个对话的原始索引
    batch = []
    batch_indices = []
    
    for idx, conversation in enumerate(chunk):
        batch.append(conversation)
        batch_indices.append(start_idx + idx)
        
        if len(batch) >= batch_size or conversation == chunk[-1]:
            # 处理当前批次
            batch_results = process_conversation_batch(batch, model_wrapper, pbar)
            # 将结果放入队列
            result_queue.put((batch_results, batch_indices))
            # 清空批次
            batch = []
            batch_indices = []
    
    pbar.close()
    # 发送完成信号
    result_queue.put(("DONE", gpu_id))

@app.command()
def main(
    *,
    input_filename: Annotated[str, typer.Option("--input-filename")] = "ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json",
    output_filename: Annotated[str, typer.Option("--output-filename")] = "mistral-parallel.json",
    output_dir: Annotated[str, typer.Option("--output-dir")] = "my_model/output",
    model_name: Annotated[str, typer.Option("--model-name")] = "Medusa/mistralai/Mistral-7B-Instruct-v0.2",
    save_interval: Annotated[int, typer.Option("--save-interval")] = 1,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 16,
    num_gpus: Annotated[int, typer.Option("--num-gpus")] = None,
):
    """主函数：使用多GPU并行处理对话"""
    # 检测可用的GPU
    available_gpus = torch.cuda.device_count()
    if num_gpus is None:
        num_gpus = available_gpus
    elif num_gpus > available_gpus:
        print(f"警告：请求的GPU数量({num_gpus})超过可用数量({available_gpus})，将使用所有可用GPU")
        num_gpus = available_gpus
    
    print(f"使用 {num_gpus} 个GPU进行并行推理")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # 加载已有结果和输入文件
    if os.path.exists(output_path):
        try:
            # 读取所有行并在首尾添加方括号使其成为有效的 JSON 数组
            with open(output_path, "r") as f:
                content = f.read()
                if content.strip():  # 确保文件不为空
                    json_str = "[" + content.rstrip().rstrip(",") + "]"
                    saved_data = json.loads(json_str)
                    processed_indices = [item["index"] for item in saved_data]  # 从每个元素中提取 index
                    # print(processed_indices)
                    print(f"加载已有处理结果: {len(processed_indices)} 条对话")
                else:
                    processed_indices = []
            
            # 找到未处理的对话
            with open(input_filename, "r") as f:
                input_data = json.loads(f.read())
            conversations = [fix_source(source["conversations"]) for source in input_data]
            conversations = [conv for i, conv in enumerate(conversations)
                           if i not in set(processed_indices)]  # 这里仍然使用set来加速查找
        except Exception as e:
            print(f"加载已有结果失败: {e}")
            with open(input_filename, "r") as f:
                input_data = json.loads(f.read())
            conversations = [fix_source(source["conversations"]) for source in input_data]
    else:
        with open(input_filename, "r") as f:
            input_data = json.loads(f.read())
        conversations = [fix_source(source["conversations"]) for source in input_data]
    
    # 根据GPU数量划分数据
    chunk_size = len(conversations) // num_gpus
    if chunk_size == 0:
        chunk_size = 1

    conversation_chunks = []
    for i in range(num_gpus - 1):
        conversation_chunks.append(conversations[i * chunk_size:(i + 1) * chunk_size])
    # 最后一个chunk包含剩余所有数据
    conversation_chunks.append(conversations[(num_gpus - 1) * chunk_size:])
    
    # 创建Manager来共享数据
    with Manager() as manager:
        shared_results = manager.list()
        shared_indices = manager.list()
        # 创建结果队列
        result_queue = manager.Queue()
        
        # 准备进程参数（移到这里）
        chunks_with_args = [
            (chunk, i, model_name, batch_size, save_interval, output_path, i * chunk_size, result_queue) 
            for i, chunk in enumerate(conversation_chunks)
        ]
        
        # 创建进程池并启动处理
        with Pool(num_gpus) as pool:
            processes = [pool.apply_async(process_chunk, (args,)) for args in chunks_with_args]
            
            # 计数器，记录完成的GPU数量
            completed_gpus = 0
            
            # 持续从队列获取结果
            while completed_gpus < num_gpus:
                result = result_queue.get()
                
                if result[0] == "DONE":
                    completed_gpus += 1
                    continue
                
                batch_results, batch_indices = result
                shared_results.extend(batch_results)
                shared_indices.extend(batch_indices)
                
                # 每处理save_interval个批次后保存
                if len(shared_results) >= save_interval * batch_size:
                    # 将新的结果转换为字典列表格式
                    new_conversations = [
                        {
                            "index": idx,
                            "conversations": conv
                        }
                        for idx, conv in zip(shared_indices, shared_results)
                    ]
                    
                    # 直接追加到文件末尾
                    with open(output_path, "a") as f:
                        for conv in new_conversations:
                            f.write(json.dumps(conv, indent=4) + "," + "\n")
                    
                    # 更新总结果，清空临时结果
                    shared_results[:] = []
                    shared_indices[:] = []
                
        print(f"\n处理完成:")
        print(f"总共处理对话数: {len(conversations)}")
        print(f"输出文件保存至: {output_path}")

if __name__ == "__main__":
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn', force=True)
    app()


    # 用这条指令： python new_data_parallel.py --num-gpus 4 --batch-size 16