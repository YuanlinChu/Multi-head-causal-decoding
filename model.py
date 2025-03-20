import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM, LlamaRMSNorm
from utils import *
from kv_cache import initialize_past_key_values
from transformers import AutoTokenizer
import os
from huggingface_hub import hf_hub_download


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
            
        base_model = KVLlamaForCausalLM.from_pretrained(
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
        print("MHC model structure:", model)
        # 从本地或远程hugging face下载 MHC 的权重文件，目前还没有传到hugging face
        mhc_head_path = os.path.join(mhc_head_name_or_path, "mhc_head.pt")
        if os.path.exists(mhc_head_path):
            filename = mhc_head_path
        else:
            filename = hf_hub_download(mhc_head_name_or_path, "mhc_head.pt")
        mhc_head_state_dict = torch.load(filename, map_location=base_model.device)
        model.mhc_head.load_state_dict(mhc_head_state_dict, strict=False)

        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):
        """Forward pass of the MHCModel.

        Each MHC head takes concatenated input from:
        1. RMSNorm(previous head's hidden states)
        2. RMSNorm(embedding of the token predicted by previous head)
        
        For the first head, we use the base model's output and its predicted token.
        """
        with torch.no_grad():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        
        # Get initial logits from base model
        current_logits = self.base_model.lm_head(hidden_states)
        
        # 初始化RMSNorm层并确保它们在正确的设备上
        device = hidden_states.device
        hidden_rmsnorm = LlamaRMSNorm(self.hidden_size).to(device)
        embed_rmsnorm = LlamaRMSNorm(self.hidden_size).to(device)
        
        mhc_logits = []
        prev_hidden = hidden_states
        
        for i in range(self.mhc_num_heads):
            # Get predicted token from previous step
            pred_token = current_logits.argmax(dim=-1)
            # Get embedding for predicted token
            token_embed = self.base_model.model.embed_tokens(pred_token)
            
            # Apply RMSNorm to both inputs
            norm_hidden = hidden_rmsnorm(prev_hidden)
            norm_embed = embed_rmsnorm(token_embed)
            
            # Concatenate normalized inputs
            combined_input = torch.cat([norm_hidden, norm_embed], dim=-1)
            
            # Pass through MHC head
            mhc_hidden_states = self.mhc_head[i](combined_input)
            current_logits = self.base_model.lm_head(mhc_hidden_states)
            mhc_logits.append(current_logits)
            
            # Update previous hidden states
            prev_hidden = mhc_hidden_states

        if output_orig:
            return torch.stack(mhc_logits, dim=0), outputs, orig
        return torch.stack(mhc_logits, dim=0)

    def mhc_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        top_k=3,  # Number of candidates per head
        max_candidates=8,  # Maximum number of sequences to keep
        posterior_threshold=0.09,
    ):
        """
        Generate text using MHC heads in a beam-search like manner.
        
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float): Temperature for sampling.
            max_steps (int): Maximum number of generation steps.
            top_k (int): Number of top candidates to consider per head.
            max_candidates (int): Maximum number of candidate sequences to maintain.
            posterior_threshold (float): Threshold for posterior validation.
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        input_ids = input_ids.clone()

        # Initialize the past key and value states
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
        
        # Get initial logits from base model
        with torch.no_grad():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            hidden_states = outputs[0]
            current_logits = self.base_model.lm_head(hidden_states)

        # Initialize candidates with just the input sequence
        candidates = [(input_ids, 0.0, hidden_states)]  # (sequence, log_prob, hidden_states)

        for idx in range(max_steps):
            new_candidates = []
            
            # Process each candidate through all MHC heads
            for sequence, log_prob, prev_hidden in candidates:
                # Get predicted token from previous step
                pred_token = sequence[:, -1:]
                
                # Process through each MHC head
                current_hidden = prev_hidden
                for head_idx in range(self.mhc_num_heads):
                    # Get token embedding
                    token_embed = self.base_model.model.embed_tokens(pred_token)
                    
                    # Concatenate and process through MHC head
                    combined_input = torch.cat([current_hidden, token_embed], dim=-1)
                    mhc_hidden = self.mhc_head[head_idx](combined_input)
                    head_logits = self.base_model.lm_head(mhc_hidden)
                    
                    # Get top-k predictions
                    if temperature == 0:
                        topk_logits, topk_indices = head_logits.topk(top_k, dim=-1)
                    else:
                        probs = F.softmax(head_logits / temperature, dim=-1)
                        topk_probs, topk_indices = probs.topk(top_k, dim=-1)
                        topk_logits = torch.log(topk_probs)
                    
                    # Create new candidates for each top-k prediction
                    for k in range(top_k):
                        new_seq = torch.cat([sequence, topk_indices[:, :, k]], dim=1)
                        new_log_prob = log_prob + topk_logits[:, :, k].item()
                        new_candidates.append((new_seq, new_log_prob, mhc_hidden))
            
            # Keep only the top max_candidates sequences
            candidates = sorted(new_candidates, key=lambda x: x[1], reverse=True)[:max_candidates]
            
            # Yield the current best candidate
            best_sequence = candidates[0][0]
            yield {
                "text": self.tokenizer.decode(
                    best_sequence[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            # Check if the best candidate has generated an EOS token
            if self.tokenizer.eos_token_id in best_sequence[0, input_len:]:
                break
