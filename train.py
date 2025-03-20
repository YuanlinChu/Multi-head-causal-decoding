# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
from safetensors.torch import save_file

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import os
from model import MHCModel, MHCConfig
from default_config import DEFAULT_CONFIG
import sys

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Customized for training MHC heads
class CustomizedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the training loss for the model.

        Args:
            model (torch.nn.Module): The model for which to compute the loss.
            inputs (dict): The input data, including input IDs, attention mask, and labels.
            return_outputs (bool): Whether to return model outputs along with the loss.
            num_items_in_batch (int, optional): Number of items in the current batch.

        Returns:
            Union[float, Tuple[float, torch.Tensor]]: The computed loss, optionally with model outputs.
        """
        # DDP will give us model.module
        if hasattr(model, "module"):
            mhc_num_heads = model.module.mhc_num_heads
        else:
            mhc_num_heads = model.mhc_num_heads

        # 获取模型输出
        logits = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        labels = inputs["labels"]
        # print(inputs)
        
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct = CrossEntropyLoss()
        log = {}
        
        # 对每个MHC头计算损失
        for i in range(mhc_num_heads):

            mhc_logits = logits[i]
            
            # 调整logits和labels以匹配MHC的预测模式
            mhc_logits = mhc_logits[:, :-(2 + i),:].contiguous()
            mhc_labels = labels[..., 2 + i:].contiguous()
            
            # 重塑张量以便计算损失
            mhc_logits = mhc_logits.view(-1, logits.shape[-1])
            mhc_labels = mhc_labels.view(-1)
            mhc_labels = mhc_labels.to(mhc_logits.device)
            
            # 计算当前头的损失
            loss_i = loss_fct(mhc_logits, mhc_labels)
            loss += loss_i

            # 过滤掉忽略标签
            not_ignore = mhc_labels.ne(IGNORE_TOKEN_ID)
            valid_labels = mhc_labels[not_ignore]
            
            # 计算top-k准确率
            if len(valid_labels) > 0:  # 确保有有效标签
                for k in range(1, 2):
                    _, topk = mhc_logits[not_ignore].topk(k, dim=-1)
                    correct = topk.eq(valid_labels.unsqueeze(-1)).any(-1)
                    log[f"mhc{i}_top{k}"] = correct.float().mean().item()
            else:
                log[f"mhc{i}_top1"] = 0.0
                
            log[f"mhc{i}_loss"] = loss_i.item()
            
        self.log(log)
        return (loss, logits) if return_outputs else loss


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # report_to: Optional[str] = None
    report_to: str = field(default="wandb")  # 明确指定使用wandb
    run_name: str = field(default="mhc_train-first_version")  
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mhc_num_heads: int = field(
        default=1,
        metadata={"help": "Number of mhc heads."},
    )
    res_layer_nums: int = field(
        default=1,
        metadata={"help": "Number of layers for each mhc head."},
    )

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    Save the model's state dictionary to a specified directory.

    Args:
        trainer (transformers.Trainer): The Hugging Face Trainer object.
        output_dir (str): The directory where the model state dictionary will be saved.
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Preprocesses conversation data and tokenizes it for model input.

    Args:
        sources: A list of conversation sources.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

    Returns:
        Dict: A dictionary containing tokenized inputs, labels, and attention mask.
    """

    # Apply prompt templates
    conversations = []
    prompts = []
    # # import pdb; pdb.set_trace()
    for i, conversation in enumerate(sources):
        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        prompts.append(prompt)
        conversations.append(conversation)

    # 打印输入数据的样本
    # print("Sample sources:", sources[:2])  # 打印前两个样本

    # Tokenize conversations
    encoding = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )

    # 打印编码后的数据
    # print("Encoded input IDs:", encoding.input_ids)
    # print("Encoded attention mask:", encoding.attention_mask)

    # Set everything to be ignored, except the assistant part
    targets = torch.full_like(encoding.input_ids, IGNORE_TOKEN_ID)
    input_ids = encoding.input_ids

    # Mask targets. Only compute loss on the assistant outputs.
    for conv_index, (conversation, target, prompt) in enumerate(zip(conversations, targets, prompts)):

        for turn in conversation:
            if turn["role"] == "assistant":
                content = turn["content"]
                # Unfortunate strip() necessary because chat templates are doing the same.
                start = prompt.index(content.strip())
                stop = start + len(content)
                indices= []
                for tok_index, (tok_start, tok_stop) in enumerate(encoding.offset_mapping[conv_index]):
                    if tok_stop >= start or tok_start < tok_stop:
                        indices.append(tok_index)
                target[indices] = encoding.input_ids[conv_index][indices]


    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = raw_data
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Lazy dataset for supervised fine-tuning.

    This dataset loads data on-the-fly when requested, which can be memory-efficient but slower.

    Args:
        raw_data (list): A list of raw data examples.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
    """

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for data preprocessing.
        data_args: Data arguments.

    Returns:
        dict: A dictionary containing train and eval datasets.
    """
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    # 打印当前工作目录和DeepSpeed配置路径，调试用
    # import os
    # print("Current working directory:", os.getcwd())
    # print("DeepSpeed config path:", os.path.abspath("deepspeed.json"))

    # 首先加载默认配置
    model_args = ModelArguments(**DEFAULT_CONFIG["model_args"])
    data_args = DataArguments(**DEFAULT_CONFIG["data_args"])
    training_args = TrainingArguments(**DEFAULT_CONFIG["training_args"])
    
    # 如果命令行有参数，则覆盖默认值
    if len(sys.argv) > 1:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    # 如果需要的上下文长度超过原始长度，设置 RoPE 缩放因子
    # 关闭模型缓存以节省内存
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    # print(config)

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False


    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    # 打印模型结构
    # print("Base model structure:", model)

    # 检查模型的权重是否为空
    # for name, param in model.named_parameters():
    #     if param.data.numel() == 0:
    #         print(f"权重为空: {name}")
    #     else:
    #         print(f"权重加载成功: {name}，形状: {param.data.shape}")

    # print(hasattr(model.model.layers[0].self_attn, "q_proj"))
    # layer = model.model.layers[0].self_attn.q_proj
    # print(f"Weights for q_proj: {layer.weight.data}")

    # 加载 tokenizer   加载预训练的分词器，设置填充标记
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token = tokenizer.eos_token

    # Making sure the tokenizer works before loading the model.
    # print(tokenizer(["This is a test", "secondary"], padding=True))
    # print(tokenizer.apply_chat_template([{"role": "user", "content": "This is a test"}],chat_template=None))

    # Freeze the base model
    for param in model.base_model.parameters():
        param.requires_grad = False

    # Add MHC heads
    mhc_heads = MHCModel(
        model,
        mhc_num_heads=training_args.mhc_num_heads,
        res_layer_nums=training_args.res_layer_nums,
        base_model_name_or_path=model_args.model_name_or_path,
    )

    # 打印模型结构
    # print("MHC model structure:", mhc_heads)
    # mhc_heads.to("cuda:0")

    # Format output dir
    training_args.output_dir = f"{training_args.output_dir}_mhc_{model_args.model_name_or_path.split('/')[-1]}_headsnum_{training_args.mhc_num_heads}_lr_{training_args.learning_rate}_layersnum_{training_args.res_layer_nums}"

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Generate mhc config for pushing to HF hub
    mhc_config = MHCConfig(
        mhc_num_heads=training_args.mhc_num_heads,
        res_layer_nums=training_args.res_layer_nums,
        base_model_name_or_path=model_args.model_name_or_path,
    )

    # Save mhc config
    mhc_config.save_pretrained(training_args.output_dir)

    # Start trainner
    trainer = CustomizedTrainer(
        model=mhc_heads, tokenizer=tokenizer, args=training_args, **data_module
    )

    # 训练和保存模型
    # 检查是否存在检查点，如果存在则恢复训练，否则从头开始训练
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    # Save mhcHead seperately
    if hasattr(mhc_heads, "module"):
        lm_head = mhc_heads.module.mhc_head
    else:
        lm_head = mhc_heads.mhc_head
        
    # 确保正确处理DeepSpeed参数
    if training_args.deepspeed:
        # 使用DeepSpeed的API正确收集分布式参数
        import deepspeed
        with deepspeed.zero.GatheredParameters(list(lm_head.parameters()), modifier_rank=0):
            if local_rank == 0:
                state_dict = lm_head.state_dict()
                # 保存模型
                tokenizer.encode("Test", truncation=None, padding="do_not_pad")
                tokenizer.save_pretrained(training_args.output_dir)
                save_file(
                    state_dict,
                    os.path.join(training_args.output_dir, "mhc_heads.safetensors"),
                )
    else:
        # 非DeepSpeed模式下的原始保存逻辑
        state_dict = lm_head.state_dict()
        # Save Medusa heads
        if local_rank == 0:
            # Modify the tokenizer internal state before saving.
            tokenizer.encode("Test", truncation=None, padding="do_not_pad")
            tokenizer.save_pretrained(training_args.output_dir)
            save_file(
                state_dict,
                os.path.join(training_args.output_dir, "mhc_heads.safetensors"),
            )


if __name__ == "__main__":
    train()
