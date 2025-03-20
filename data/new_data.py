import typer
import json
from typing_extensions import Annotated
import tqdm
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import os

app = typer.Typer()

class ModelWrapper:
    def __init__(self, model_name="Medusa/mistralai/Mistral-7B-Instruct-v0.2"):
        # 初始化模型和分词器，使用本地路径
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True
        )
        # 设置填充标记为 EOS 标记
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
            # 添加以下参数来优化性能
            use_cache=True,
            low_cpu_mem_usage=True,
        )
        # 设置为评估模式
        self.model.eval()
        
        # 设置最大长度限制
        self.max_input_length = 4000
        self.max_total_tokens = 4096
        self.max_batch_prefill_tokens = 4000
    
    @torch.inference_mode()  # 使用推理模式装饰器
    def generate_response(self, messages):
        # 将消息格式化为 Mistral 格式
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"[INST] {msg['content']} [/INST]"
            else:
                prompt += msg['content']
        
        # 对输入进行截断，确保不超过最大输入长度
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                 max_length=self.max_input_length).to(self.model.device)
        
        # 设置生成参数以匹配目标模型的推理分布
        outputs = self.model.generate(
            **input_ids,
            temperature=0.7,  # 保持适中的温度，这是大多数LLM默认的设置
            do_sample=True,   # 启用采样
            top_p=0.95,       # 使用较高的top_p值，保留更多的概率分布
            top_k=0,          # 禁用top_k采样，完全依赖核采样
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_total_tokens,
            min_length=1,
            # max_new_tokens=512,
            use_cache=True,
            repetition_penalty=1.0,  # 移除重复惩罚，保持原始分布
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取最后一个回复
        last_response = response.split("[/INST]")[-1].strip()
        
        return {"role": "assistant", "content": last_response}

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

def recreate_conversation(conversation, model_wrapper):
    """处理单个对话的所有轮次"""
    messages = []
    try:
        # 获取所有用户消息
        user_messages = [msg for msg in conversation if msg["role"] == "user"]
        
        # 逐个处理用户消息
        for message in user_messages:
            messages.append(message)
            response = model_wrapper.generate_response(messages.copy())
            messages.append(response)
    except Exception as e:
        print(f"对话处理失败: {e}")
        return None
    
    return messages

def process_conversation_batch(batch, model_wrapper, pbar):
    """真正的批量处理对话"""
    all_results = []
    
    # 收集每个对话当前的状态
    current_states = [([], conv) for conv in batch]  # (历史消息, 剩余对话)
    
    # 为每个对话创建一个历史记录列表
    conversation_histories = [[] for _ in batch]
    
    # 持续处理，直到所有对话都完成
    while current_states:
        # 准备这一批次的输入
        batch_inputs = []
        active_indices = []
        
        for i, (history, remaining) in enumerate(current_states):
            # 获取下一个用户消息
            user_messages = [msg for msg in remaining if msg["role"] == "user"]
            if not user_messages:
                continue
                
            # 准备输入（历史 + 新的用户消息）
            current_messages = history + [user_messages[0]]
            batch_inputs.append(current_messages)
            active_indices.append(i)
            
        if not batch_inputs:
            break
            
        # 批量生成回复
        responses = model_wrapper.batch_generate_response(batch_inputs)
        
        # 更新每个活跃对话的状态
        for idx, response in zip(active_indices, responses):
            history, remaining = current_states[idx]
            
            # 更新历史
            user_msg = [msg for msg in remaining if msg["role"] == "user"][0]
            new_history = history + [user_msg, response]
            
            # 更新对话历史记录
            conversation_histories[idx].extend([user_msg, response])
            
            # 更新剩余对话
            new_remaining = [msg for msg in remaining[remaining.index(user_msg)+1:]]
            
            if new_remaining:
                current_states[idx] = (new_history, new_remaining)
            else:
                # 对话完成，保存结果
                all_results.append(conversation_histories[idx])
                current_states[idx] = ([], [])
                
        # 移除已完成的对话
        current_states = [(h, r) for h, r in current_states if r]
        
        # 更新进度
        pbar.update(len(responses))
    
    # 确保所有完成的对话都被添加到结果中
    for history in conversation_histories:
        if history and history not in all_results:
            all_results.append(history)
    
    return all_results

@app.command()
def main(
    *,
    input_filename: Annotated[str, typer.Option("--input-filename")] = "Medusa/ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json",
    output_filename: Annotated[str, typer.Option("--output-filename")] = "mistral.json",
    output_dir: Annotated[str, typer.Option("--output-dir")] = "my_model/output",
    model_name: Annotated[str, typer.Option("--model-name")] = "Medusa/mistralai/Mistral-7B-Instruct-v0.2",
    save_interval: Annotated[int, typer.Option("--save-interval")] = 1,  # 每处理1个批次保存一次
    batch_size: Annotated[int, typer.Option("--batch-size")] = 16,  # 每批处理的对话数
):
    """主函数：加载模型并批量处理对话"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # 如果输出文件已存在，尝试加载已有的处理结果
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                recreated_conversations = json.load(f)
            print(f"加载已有处理结果: {len(recreated_conversations)} 条对话")
            start_idx = len(recreated_conversations)
        except:
            recreated_conversations = []
            start_idx = 0
    else:
        recreated_conversations = []
        start_idx = 0
    
    print("正在加载模型...")
    model_wrapper = ModelWrapper(model_name)
    print("模型加载完成！")
    
    # 读取输入文件
    with open(input_filename, "r") as f:
        input_data = json.loads(f.read())
    conversations = [fix_source(source["conversations"]) for source in input_data]
    
    # 只处理未处理的对话
    conversations = conversations[start_idx:]
    
    # 计算总对话轮次
    total_turns = sum(len([m for m in conv if m["role"] == "user"]) for conv in conversations)
    
    # 创建总体进度条
    main_pbar = tqdm(
        total=len(conversations),
        desc="总体进度",
        position=0,
        leave=True
    )
    
    # 创建对话轮次进度条
    turns_pbar = tqdm(
        total=total_turns,
        desc="对话轮次",
        position=1,
        leave=True
    )

    success_count = 0
    
    # 处理对话
    batch = []
    batch_count = 0
    
    for conversation in conversations:
        batch.append(conversation)
        
        # 当批次满了或是最后一个对话时，进行处理
        if len(batch) >= batch_size or conversation == conversations[-1]:
            results = process_conversation_batch(batch, model_wrapper, turns_pbar)
            recreated_conversations.extend(results)
            # print(recreated_conversations)
            success_count += len(results)
            main_pbar.update(len(batch))  # 更新总体进度
            
            batch_count += 1
            # 每处理save_interval个批次保存一次
            if batch_count % save_interval == 0:
                with open(output_path, "w") as f:
                    json.dump(recreated_conversations, f, indent=4)
            
            # 清空批次
            batch = []
    
    # 关闭进度条
    main_pbar.close()
    turns_pbar.close()

    total = len(conversations)
    print(f"\n处理完成:")
    print(f"总共对话数: {total}")
    print(f"成功处理: {success_count}")
    print(f"失败数量: {total - success_count}")
    print(f"总对话轮次: {total_turns}")
    print(f"输出文件保存至: {output_path}")

    # 最后保存一次
    with open(output_path, "w") as f:
        json.dump(recreated_conversations, f, indent=4)

if __name__ == "__main__":
    app()