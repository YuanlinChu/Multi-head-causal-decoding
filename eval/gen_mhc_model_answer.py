"""Generate answers with local models.

Usage:
python gen_mhc_model_answer.py --model-path  --model-id 
"""

import argparse
import json
import os
import random
import time
import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template

# mhc imports
import transformers

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import *
from model import MHCModel
from kv_cache import initialize_past_key_values
from mhc_choices import *

def mhc_forward_time(input_ids, model, tokenizer, mhc_choices, temperature, posterior_threshold, posterior_alpha, max_steps = 512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()

    # 对路径进行排序，确保依赖关系正确处理
    node_paths = sorted(mhc_choices, key=lambda x: (len(x), x))

    # 缓存MHC buffers（用于树形注意力的固定模式）
    if hasattr(model, "node_paths") and model.node_paths == node_paths:
        # 加载缓存的MHC buffer
        mhc_buffers = model.mhc_buffers
    else:
        # 初始化MHC buffer
        mhc_buffers = generate_mhc_buffers(
            node_paths, device=model.base_model.device
        )
    model.mhc_buffers = mhc_buffers
    model.node_paths = node_paths

    # 初始化past key和value状态
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]

    reset_mhc_mode(model)  # 改为reset_mhc_mode
    # 初始化树形注意力掩码并处理预填充tokens
    base_hidden, logits = initialize_mhc(  # 改为initialize_mhc
        input_ids, model, mhc_buffers["mhc_attn_mask"], past_key_values, mhc_choices
    )

    new_token = 0
    last_round_token = 0
    
    # 初始化时间统计变量
    total_times = []
    generate_times = []
    tree_decoding_times = []
    evaluate_times = []
    update_times = []
    accept_lengths = []
    
    # 初始化best_candidate统计
    candidate_counts = {}
    # 不再预先初始化固定数量的键
    
    for idx in range(max_steps):
        # 记录整个循环的开始时间
        loop_start_time = time.time()
        
        # 使用MHC头的topk预测生成候选项
        start_time = time.time()
        candidates, flatten_candidates, path_logits = generate_candidates(
            model,
            base_hidden, 
            logits,
            mhc_buffers["retrieve_indices"],
            mhc_buffers["flatten_len"],
            node_paths,
        )
        generate_time = time.time() - start_time
        
        # 使用树形注意力验证候选项并获取预测
        start_time = time.time()
        outputs, logits = tree_decoding(
            model,
            flatten_candidates,
            past_key_values,
            mhc_buffers["mhc_position_ids"],
            input_ids,
            mhc_buffers["retrieve_indices"],
        )
        tree_decoding_time = time.time() - start_time
        
        # 评估候选项的后验概率以选择接受的候选前缀
        start_time = time.time()
        best_candidate, accept_length = evaluate_posterior(
            logits, candidates, temperature, posterior_threshold, posterior_alpha
        )
        evaluate_time = time.time() - start_time
        
        # 统计best_candidate的分布 - 修复张量处理
        if accept_length == 0:
            best_candidate_key = -1  # 当accept_length为0时，标记为-1
        else:
            best_candidate_key = int(best_candidate.item()) if torch.is_tensor(best_candidate) else best_candidate
        
        if best_candidate_key not in candidate_counts:
            candidate_counts[best_candidate_key] = 0
        candidate_counts[best_candidate_key] += 1
        
        # 更新input_ids和logits
        start_time = time.time()
        input_ids, logits, new_token, base_hidden = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            mhc_buffers["retrieve_indices"],
            outputs,
            logits,
            new_token,
            past_key_values_data,
            current_length_data,
        )
        update_time = time.time() - start_time
        
        # 计算整个循环的总时间
        loop_total_time = time.time() - loop_start_time
        
        # 打印每个步骤的时间信息
        # print(f"Step {idx} - Total: {loop_total_time:.4f}s | Generate: {generate_time:.4f}s ({generate_time/loop_total_time*100:.1f}%) | "
        #       f"TreeDecode: {tree_decoding_time:.4f}s ({tree_decoding_time/loop_total_time*100:.1f}%) | "
        #       f"Evaluate: {evaluate_time:.4f}s ({evaluate_time/loop_total_time*100:.1f}%) | "
        #       f"Update: {update_time:.4f}s ({update_time/loop_total_time*100:.1f}%) | "
        #       f"Accept Length: {accept_length}")

        # 收集时间统计数据
        total_times.append(loop_total_time)
        generate_times.append(generate_time)
        tree_decoding_times.append(tree_decoding_time)
        evaluate_times.append(evaluate_time)
        update_times.append(update_time)
        accept_lengths.append(accept_length)

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
    
    # 计算并输出平均时间统计
    if len(total_times) > 0:
        avg_total = sum(total_times) / len(total_times)
        avg_generate = sum(generate_times) / len(generate_times)
        avg_tree_decoding = sum(tree_decoding_times) / len(tree_decoding_times)
        avg_evaluate = sum(evaluate_times) / len(evaluate_times)
        avg_update = sum(update_times) / len(update_times)
        avg_accept_length = sum(accept_lengths) / len(accept_lengths)
        
        print(f"生成统计 ({idx+1} 步) - 平均每步时间: {avg_total:.4f}s | "
              f"生成: {avg_generate:.4f}s ({avg_generate/avg_total*100:.1f}%) | "
              f"树解码: {avg_tree_decoding:.4f}s ({avg_tree_decoding/avg_total*100:.1f}%) | "
              f"评估: {avg_evaluate:.4f}s ({avg_evaluate/avg_total*100:.1f}%) | "
              f"更新: {avg_update:.4f}s ({avg_update/avg_total*100:.1f}%) | "
              f"平均接受长度: {avg_accept_length:.2f}")
        
        # 输出best_candidate分布统计
        total_candidates = sum(candidate_counts.values())
        print("\nbest_candidate 分布统计:")
        for candidate_idx, count in sorted(candidate_counts.items()):
            if total_candidates > 0:
                percentage = (count / total_candidates) * 100
                if candidate_idx == -1:
                    path_info = "拒绝所有候选 (accept_length=0)"
                else:
                    path_info = f"路径 {candidate_idx}"
                    if candidate_idx < len(node_paths):
                        path_info += f": {node_paths[candidate_idx]}"
                print(f"  {path_info} - 次数: {count} ({percentage:.2f}%)")
    
    return input_ids, new_token, idx


def mhc_forward(input_ids, model, tokenizer, mhc_choices, temperature, posterior_threshold, posterior_alpha, max_steps = 512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()

    # 对路径进行排序，确保依赖关系正确处理
    node_paths = sorted(mhc_choices, key=lambda x: (len(x), x))

    # 缓存MHC buffers（用于树形注意力的固定模式）
    if hasattr(model, "node_paths") and model.node_paths == node_paths:
        # 加载缓存的MHC buffer
        mhc_buffers = model.mhc_buffers
    else:
        # 初始化MHC buffer
        mhc_buffers = generate_mhc_buffers(
            node_paths, device=model.base_model.device
        )
    model.mhc_buffers = mhc_buffers
    model.node_paths = node_paths

    # 初始化past key和value状态
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]

    reset_mhc_mode(model)  # 改为reset_mhc_mode
    # 初始化树形注意力掩码并处理预填充tokens
    base_hidden, logits = initialize_mhc(  # 改为initialize_mhc
        input_ids, model, mhc_buffers["mhc_attn_mask"], past_key_values, mhc_choices
    )

    new_token = 0

    for idx in range(max_steps):
        # 使用MHC头的topk预测生成候选项
        candidates, flatten_candidates, path_logits = generate_candidates(
            model,
            base_hidden, 
            logits,
            mhc_buffers["retrieve_indices"],
            mhc_buffers["flatten_len"],
            node_paths,
        )

        # 使用树形注意力验证候选项并获取预测
        outputs, logits = tree_decoding(
            model,
            flatten_candidates,
            past_key_values,
            mhc_buffers["mhc_position_ids"],
            input_ids,
            mhc_buffers["retrieve_indices"],
        )

        # 评估候选项的后验概率以选择接受的候选前缀
        best_candidate, accept_length = evaluate_posterior(
            logits, candidates, temperature, posterior_threshold, posterior_alpha
        )

        # 更新input_ids和logits
        input_ids, logits, new_token, base_hidden = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            mhc_buffers["retrieve_indices"],
            outputs,
            logits,
            new_token,
            past_key_values_data,
            current_length_data,
        )

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
    return input_ids, new_token, idx

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    posterior_threshold,
    posterior_alpha,
    mhc_choices,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model) # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                posterior_threshold,
                posterior_alpha,
                mhc_choices,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    temperature,
    posterior_threshold,
    posterior_alpha,
    mhc_choices,
):
    
    # Medusa model setup
    num_heads = 5

    model = MHCModel.from_pretrained(
        model_path,
        mhc_num_heads = num_heads,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    
    model.eval()
    print('Check model training state:',model.training)
    
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)
    
    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)
        conv = get_conversation_template(model_id)
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            # if temperature < 1e-4:
            #     do_sample = False
            # else:
            #     do_sample = True

            # some models may error out when generating long outputs
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx = mhc_forward_time(
                    torch.as_tensor(input_ids).cuda(),
                    model,
                    tokenizer,
                    mhc_choices,
                    temperature,
                    posterior_threshold,
                    posterior_alpha,
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time
                output_ids = output_ids[0][len(input_ids[0]) :]
                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print("ERROR question ID: ", question["question_id"])
                print(f"错误详情: {str(e)}")
                output = "ERROR"

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')


    for question in tqdm(questions):
        if question["category"] in temperature_config:
            temperature = temperature_config[question["category"]]
        else:
            temperature = 0.7

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template(model_id)
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                # if temperature < 1e-4:
                #     do_sample = False
                # else:
                #     do_sample = True

                # some models may error out when generating long outputs
                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, idx = mhc_forward(
                        torch.as_tensor(input_ids).cuda(),
                        model,
                        tokenizer,
                        mhc_choices,
                        temperature,
                        posterior_threshold,
                        posterior_alpha,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    # if model.config.is_encoder_decoder:
                    #     output_ids = output_ids[0]
                    # else:
                    output_ids = output_ids[0][len(input_ids[0]) :]

                    # be consistent with the template's stop_token_ids
                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="/hpc2hdd/home/ychu763/Documents/my_model/shareGPT-4epochs_mhc_vicuna-7b-v1.3_headsnum_5_lr_0.001_layersnum_1",
        # required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", 
        type=str, 
        default="vicuna-mhc-7b-v1.3-shareGPT",
        # required=True,
        )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    # YL: Medusa args
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for mhc sampling.",
    )

    parser.add_argument(
        "--posterior-threshold",
        type=float,
        default=0.09,
        help="The posterior threshold for mhc sampling.",
    )
    
    parser.add_argument(
        "--posterior-alpha",
        type=float,
        default=0.3,
        help="The posterior alpha for mhc sampling.",
    )

    parser.add_argument(
        "--mhc-choices",
        type=str,
        default="mc_sim_7b_63",
        help="The mhc choices for mhc sampling.",
    )




    args = parser.parse_args()

    args.model_id = args.model_id+"-temperature-"+str(args.temperature)+"-posterior_threshold-"+str(args.posterior_threshold)+"-posterior_alpha-"+str(args.posterior_alpha)
    args.mhc_choices = eval(args.mhc_choices)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,

        args.temperature,
        args.posterior_threshold,
        args.posterior_alpha,
        args.mhc_choices,
    )

    reorg_answer_file(answer_file)