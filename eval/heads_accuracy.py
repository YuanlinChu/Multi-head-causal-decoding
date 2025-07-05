import os
import torch
import json
import numpy as np
from tqdm import tqdm
from model import MHCModel
import argparse
from fastchat.model.model_adapter import get_conversation_template
from kv_cache import initialize_past_key_values
from utils import reset_mhc_mode

def get_accuracies(mhc, logit):
    # 获取每个头的正确计数
    seq_len, choices, topk = mhc.shape
    results = []
    for choice in range(choices):
        results.append(mhc[:-choice - 1, choice].eq(logit[choice + 1:, 0]))
    return results

def main(args):
    # 加载模型
    model = MHCModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = model.get_tokenizer()

    # 加载数据
    data = json.load(open(args.data_path))
    
    # 初始化KV缓存
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.base_model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data
    results = None

    # 处理每个样本
    for sample in tqdm(data):
        # 准备对话模板
        conv = get_conversation_template("vicuna")
        conv.messages = []
        conv.append_message(conv.roles[0], sample["instruction"])
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        
        steps = args.steps
        logits_ids = []
        mhc_topk_ids = []

        with torch.inference_mode():
            # 编码输入
            input_ids = tokenizer([prompt]).input_ids
            input_ids = torch.as_tensor(input_ids).cuda()
            model.current_length_data.zero_()  # 重置运行
            reset_mhc_mode(model)
            
            # 第一步预测
            mhc_logits, outputs, logits = model(
                input_ids, past_key_values=past_key_values, output_orig=True
            )
            _, mhc_topk = mhc_logits[..., -1, :].topk(20, dim=-1)
            input_id = logits[:, -1:].argmax(dim=-1)
            logits_ids.append(input_id.detach().cpu())
            mhc_topk_ids.append(mhc_topk.detach().cpu())
            
            # 后续步骤预测
            for _ in range(steps):
                mhc_logits, outputs, logits = model(
                    input_id, past_key_values=past_key_values, output_orig=True
                )
                _, mhc_topk = mhc_logits[..., -1, :].topk(20, dim=-1)
                input_id = logits[:, -1:].argmax(dim=-1)
                logits_ids.append(input_id.detach().cpu())
                mhc_topk_ids.append(mhc_topk.detach().cpu())
            
            # 整理结果
            logits_ids = torch.stack(logits_ids, dim=0)
            mhc_topk_ids = torch.stack(mhc_topk_ids, dim=0).squeeze(2)
            
            if results is None:
                results = get_accuracies(mhc_topk_ids, logits_ids)
            else:
                # 合并结果
                cur_results = get_accuracies(mhc_topk_ids, logits_ids)
                for i in range(len(results)):
                    results[i] = torch.cat((results[i], cur_results[i]), dim=0)

    # 计算每个头的接受率
    head_acceptance_rates = [result.float().mean().item() for result in results]
    
    # 计算平均接受率
    avg_acceptance_rate = sum(head_acceptance_rates) / len(head_acceptance_rates)
    
    # 打印结果
    print(f"平均接受率: {avg_acceptance_rate:.4f}")
    for i, rate in enumerate(head_acceptance_rates):
        print(f"头 {i} 接受率: {rate:.4f}")
    
    # 保存结果
    save_path = os.path.join(args.save_dir, args.model_name + "_heads_accuracy.pt")
    torch.save(results, save_path)
    
    # 同时保存可读的接受率数据
    rates_path = os.path.join(args.save_dir, args.model_name + "_acceptance_rates.pt")
    torch.save(torch.tensor(head_acceptance_rates), rates_path)
    print(f"结果已保存至 {save_path} 和 {rates_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MHC模型评估器")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="预训练MHC模型的路径")
    parser.add_argument("--model_name", type=str, required=True,
                        help="模型名称")
    parser.add_argument("--data_path", type=str, required=True,
                        help="JSON格式评估数据的路径")
    parser.add_argument("--save_dir", type=str, default="data",
                        help="保存结果的目录")
    parser.add_argument("--steps", type=int, default=20,
                        help="运行模型的步数")
    
    args = parser.parse_args()
    
    # 如果保存目录不存在，创建它
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    main(args)