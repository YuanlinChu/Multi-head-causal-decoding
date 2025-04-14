import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from modeling_llama_kv import LlamaForCausalLM as KVForCausalLM
# from modeling_mistral_kv import MistralForCausalLM as KVForCausalLM
from mhc_choices import mc_sim_7b_63
from utils import *
from kv_cache import initialize_past_key_values
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class MHCConfig(PretrainedConfig):

    def __init__(
        self,
        mhc_num_heads=4,
        res_layer_nums=1,
        base_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mhc_num_heads = mhc_num_heads
        self.res_layer_nums = res_layer_nums
        self.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class MHCModel(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        base_model,
        mhc_num_heads=4,
        res_layer_nums=1,
        base_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
    ):
        """
        Args:
            base_model (nn.Module): The base language model to be used.
            medusa_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
            medusa_num_layers (int, optional): Number of ResBlock layers for each Medusa head. Defaults to 0.
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.config.hidden_size
        self.vocab_size = base_model.config.vocab_size
        self.mhc_num_heads = mhc_num_heads
        self.res_layer_nums = res_layer_nums
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Medusa heads
        self.mhc_head = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    *([ResBlock(self.hidden_size)] * res_layer_nums),
                )
                for _ in range(mhc_num_heads)
            ]
        )

        # Ensure mhc_head's dtype and device align with the base_model
        self.mhc_head.to(self.base_model.dtype).to(self.base_model.device)

        # 在初始化时创建RMSNorm实例
        self.hidden_rmsnorm = RMSNorm(self.hidden_size)
        self.embed_rmsnorm = RMSNorm(self.hidden_size)
        
        # 确保RMSNorm与base_model使用相同的设备和数据类型
        self.hidden_rmsnorm.to(self.base_model.dtype).to(self.base_model.device)
        self.embed_rmsnorm.to(self.base_model.dtype).to(self.base_model.device)        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        mhc_head_name_or_path,
        base_model=None,
        mhc_num_heads=None,
        **kwargs,
    ):
        """
        Args:
            medusa_head_name_or_path (str): Name or path of the Medusa head to load.
            **kwargs: Additional keyword arguments for loading the base model.

        Returns:
            MedusaModel: A MedusaModel instance loaded from the given path.
        """
        mhc_config = MHCConfig.from_pretrained(mhc_head_name_or_path)
        if mhc_num_heads is not None:
            print("Overriding mhc_num_heads as:", mhc_num_heads)
            mhc_config.mhc_num_heads = mhc_num_heads
        if base_model is not None:
            print("Overriding base_model as:", base_model)
            mhc_config.base_model_name_or_path = base_model
            
        base_model = KVForCausalLM.from_pretrained(
            mhc_config.base_model_name_or_path, **kwargs
        )

        # 用当前类（cls）创建了一个 MHCModel 实例
        model = cls(
            base_model,
            mhc_config.mhc_num_heads,
            mhc_config.res_layer_nums,
            mhc_config.base_model_name_or_path,
        )
        # 打印模型结构
        # print("MHC model structure:", model)
        # 从本地或远程hugging face下载 MHC 的权重文件
        mhc_head_path = os.path.join(mhc_head_name_or_path, "mhc_heads.pt")
        safetensors_path = os.path.join(mhc_head_name_or_path, "mhc_heads.safetensors")
        
        # 首先尝试加载 safetensors 格式
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            filename = safetensors_path
            mhc_head_state_dict = load_file(filename)
            mhc_head_state_dict = {k: v.to(model.base_model.device) for k, v in mhc_head_state_dict.items()}
        # 然后尝试加载 pt 格式
        elif os.path.exists(mhc_head_path):
            filename = mhc_head_path
            mhc_head_state_dict = torch.load(filename, map_location=model.device)
        # 最后尝试从 Hugging Face hub 下载
        else:
            filename = hf_hub_download(mhc_head_name_or_path, "mhc_heads.pt")
            mhc_head_state_dict = torch.load(filename, map_location=base_model.device)
        model.mhc_head.load_state_dict(mhc_head_state_dict, strict=True)
        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        inference_mode=False,
    ):
        """Forward pass of the MHCModel.

        Each MHC head takes concatenated input from:
        1. RMSNorm(previous head's hidden states)
        2. RMSNorm(embedding of the token predicted by previous head)
        
        For the first head, we use the base model's output and its predicted token.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Labels for computing the language modeling loss.
            past_key_values (tuple, optional): Tuple of past key and value states.
            output_orig (bool, optional): Whether to output the original model's output.
            position_ids (torch.Tensor, optional): Position IDs.
            inference_mode (bool, optional): Whether to run in inference mode. Default is False.
        """
        with torch.no_grad():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            orig = self.base_model.lm_head(outputs[0])
        
        if inference_mode:
            return None, outputs, orig
        else:
            # 训练模式
            # Clone the output hidden states
            hidden_states = outputs[0].clone()
            
            # Get initial logits from base model
            current_logits = self.base_model.lm_head(hidden_states)
            
            mhc_logits = []
            prev_hidden = hidden_states
            for i in range(self.mhc_num_heads):
                # if labels is not None and i+1 < labels.size(1):
                #     # true_token = labels[:, i+1:i+2]  # 获取第i+1位置的真值
                    
                #     # 计算输入序列的有效长度（非padding部分）
                #     seq_length = attention_mask.sum(dim=1, keepdim=True).to(torch.long)
                #     # 获取序列末尾后的第i+1个位置的真值
                #     true_token = labels.gather(1, seq_length + i).unsqueeze(1)
                    
                #     if i == 0:
                #         pred_token = torch.cat([
                #             input_ids[:, 1:],   # 去掉第一个token
                #             true_token          # 使用下一个位置的真值
                #         ], dim=1)
                #     else:
                #         # 后续的头使用前一个预测结果
                #         pred_token = torch.cat([
                #             pred_token[:, 1:],  # 去掉第一个token
                #             true_token          # 使用对应位置的真值
                #         ], dim=1)
                #     token_embed = self.base_model.model.embed_tokens(pred_token)
                # else:
                #     # 如果没有提供labels或已超出labels范围，则回退到使用预测的token
                #     pred_token = torch.cat([
                #         pred_token[:, 1:],   # 去掉第一个token
                #         current_logits[:, -1:].argmax(dim=-1)
                #     ], dim=1)
                #     token_embed = self.base_model.model.embed_tokens(pred_token)

                # 将-100替换为填充标记ID
                true_token = nn.functional.pad(labels[:, i+1:], (0, i+1), mode='constant', value=-100)
                
                # 检查并替换无效的token ID
                mask = true_token == -100
                if mask.any():
                    # 使用填充标记ID替换-100
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                    true_token = true_token.masked_fill(mask, pad_token_id)
                
                token_embed = self.base_model.model.embed_tokens(true_token)
                
                norm_hidden = self.hidden_rmsnorm(prev_hidden)
                norm_embed = self.embed_rmsnorm(token_embed)
                combined_input = torch.cat([norm_hidden, norm_embed], dim=-1)
                
                mhc_hidden_states = self.mhc_head[i](combined_input)
                current_logits = self.base_model.lm_head(mhc_hidden_states)
                mhc_logits.append(current_logits)
                prev_hidden = mhc_hidden_states

            if output_orig:
                return torch.stack(mhc_logits, dim=0), outputs, orig
            return torch.stack(mhc_logits, dim=0)

    def mhc_forward(self, head_idx, prev_hidden, token_idx):
        """
        执行单个MHC头的前向传播
        
        Args:
            head_idx (int): MHC头的索引
            prev_hidden (torch.Tensor): 前一个MHC头的输出隐藏层，形状为[batch_size, seq_len, hidden_size]
            token_idx (torch.Tensor): 前一个采样的token索引，形状为[batch_size, 1]
            
        Returns:
            tuple: (mhc_hidden_states, current_logits) - MHC头的输出隐藏层和对应的logits
        """
        # 将token索引转换为嵌入向量
        token_embed = self.base_model.model.embed_tokens(token_idx)
        
        # 初始化RMSNorm层并确保它们在正确的设备上
        device = prev_hidden.device
        hidden_rmsnorm = RMSNorm(self.hidden_size).to(device)
        embed_rmsnorm = RMSNorm(self.hidden_size).to(device)
        
        # 只对隐藏层的最后一个位置应用RMSNorm
        last_hidden = prev_hidden[:, -1:, :]
        norm_hidden = self.hidden_rmsnorm(last_hidden)
        norm_embed = self.embed_rmsnorm(token_embed)
        
        # 将两个向量连接在一起
        combined_input = torch.cat([norm_hidden, norm_embed], dim=-1)
        
        # 通过MHC head
        mhc_hidden_states = self.mhc_head[head_idx](combined_input)
        current_logits = self.base_model.lm_head(mhc_hidden_states)
        
        return mhc_hidden_states, current_logits

    def mhc_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        mhc_choices=mc_sim_7b_63,  # 改为mhc_choices
        posterior_threshold=0.09,  # MHC输出验证的阈值
        # 另一个阈值超参数，建议设为posterior_threshold的平方根
        posterior_alpha=0.3,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            mhc_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # 对路径进行排序，确保依赖关系正确处理
        node_paths = sorted(mhc_choices, key=lambda x: (len(x), x))

        # 缓存MHC buffers（用于树形注意力的固定模式）
        if hasattr(self, "node_paths") and self.node_paths == node_paths:
            # 加载缓存的MHC buffer
            mhc_buffers = self.mhc_buffers
        else:
            # 初始化MHC buffer
            mhc_buffers = generate_mhc_buffers(
                node_paths, device=self.base_model.device
            )
        self.mhc_buffers = mhc_buffers
        self.node_paths = node_paths

        # 初始化past key和value状态
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_mhc_mode(self)  # 改为reset_mhc_mode
        # 初始化树形注意力掩码并处理预填充tokens
        base_hidden, logits = initialize_mhc(  # 改为initialize_mhc
            input_ids, self, mhc_buffers["mhc_attn_mask"], past_key_values, mhc_choices
        )

        new_token = 0


        for idx in range(max_steps):
            # 使用MHC头的topk预测生成候选项
            candidates, flatten_candidates, path_logits = generate_candidates(
                self,
                base_hidden, 
                logits,
                mhc_buffers["retrieve_indices"],
                mhc_buffers["flatten_len"],
                node_paths,
            )

            # 使用树形注意力验证候选项并获取预测
            outputs, logits = tree_decoding(
                self,
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

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
