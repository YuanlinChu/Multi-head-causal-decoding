import torch
import torch.nn.functional as F
import time

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def generate_mhc_buffers(node_paths, device="cuda"):
    """
    根据node_paths生成MHC所需的缓冲区
    
    Parameters:
    - node_paths (list): 排序后的路径参数，每个路径是一个整数列表
    - device (str): 设备名称，默认为"cuda"
    
    Returns:
    - dict: 包含MHC相关缓冲区的字典
    """

    # 构建前缀树和路径映射
    path_to_id = {}
    for i, path in enumerate(node_paths):
        for j in range(len(path) + 1):
            prefix = tuple(path[:j])
            if prefix not in path_to_id:
                path_to_id[prefix] = len(path_to_id)
    
    # 计算展平后的长度（每个唯一前缀对应一个token）
    flatten_len = len(path_to_id)
    
    # 创建前缀到展平索引的映射
    prefix_to_flatten_idx = {}
    for prefix, idx in path_to_id.items():
        prefix_to_flatten_idx[prefix] = idx
    
    # 生成位置ID
    position_ids = torch.zeros(flatten_len, dtype=torch.long)
    for prefix, idx in prefix_to_flatten_idx.items():
        position_ids[idx] = len(prefix)
    
    # 创建注意力掩码
    mhc_attn_mask = torch.zeros(flatten_len, flatten_len)
    
    # 所有token都可以看到base token（空前缀）
    base_idx = prefix_to_flatten_idx[()]
    mhc_attn_mask[:, base_idx] = 1
    
    # 每个token可以看到自己
    for i in range(flatten_len):
        mhc_attn_mask[i, i] = 1
    
    # 为每个前缀设置注意力掩码
    for path in node_paths:
        for depth in range(1, len(path) + 1):
            prefix = tuple(path[:depth])
            if prefix in prefix_to_flatten_idx:
                token_idx = prefix_to_flatten_idx[prefix]
                
                # 该token可以看到其所有祖先
                for ancestor_depth in range(depth):
                    ancestor_prefix = tuple(path[:ancestor_depth])
                    if ancestor_prefix in prefix_to_flatten_idx:
                        ancestor_idx = prefix_to_flatten_idx[ancestor_prefix]
                        mhc_attn_mask[token_idx, ancestor_idx] = 1
    
    # 扩展注意力掩码的维度以符合transformer的要求
    mhc_attn_mask = mhc_attn_mask.unsqueeze(0).unsqueeze(0).to(device)
    mhc_position_ids = position_ids.to(device)
    
    # 创建从candidates到flatten_candidates的映射（flatten_indices）
    max_path_length = max(len(path) for path in node_paths) + 1  # +1 for base token
    flatten_indices = []
    for path in node_paths:
        # 为每个路径创建一个映射数组，索引是深度，值是flatten_candidates中的位置
        path_indices = torch.full((max_path_length,), -1, dtype=torch.long, device=device)
        for depth in range(len(path) + 1):
            prefix = tuple(path[:depth])
            if prefix in prefix_to_flatten_idx:
                path_indices[depth] = prefix_to_flatten_idx[prefix]
        flatten_indices.append(path_indices)
    
    # 将flatten_indices转换为二维张量
    retrieve_indices = torch.stack(flatten_indices, dim=0)
    
    # 聚合生成的缓冲区到字典
    mhc_buffers = {
        "mhc_attn_mask": mhc_attn_mask,
        "mhc_position_ids": mhc_position_ids,
        # "flatten_indices": flatten_indices,
        "retrieve_indices": retrieve_indices,
        # "prefix_to_flatten_idx": prefix_to_flatten_idx
        "flatten_len": flatten_len
    }
    
    return mhc_buffers

def initialize_mhc(input_ids, model, mhc_attn_mask, past_key_values, node_paths):
    """
    Initializes the Medusa structure for a given model.

    This function performs the following operations:
    1. Forward pass through the model to obtain the Medusa logits, original model outputs, and logits.
    2. Sets the Medusa attention mask within the base model.

    Args:
    - input_ids (torch.Tensor): The input tensor containing token ids.
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - mhc_attn_mask (torch.Tensor): The attention mask designed specifically for the Mhc structure.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - mhc_logits (torch.Tensor): Logits from the Medusa heads.
    - logits (torch.Tensor): Original logits from the base model.
    """
    _, outputs, logits = model(
        input_ids, past_key_values=past_key_values, inference_mode=True,
    )
    model.base_model.model.mhc_mask = mhc_attn_mask
    return outputs[0], logits


def reset_mhc_mode(
    model,
):
    """
    Resets the Medusa settings and the past key-values to their initial state.

    This function ensures that after any operations involving Medusa,
    the base model and its settings return to their default state.
    Specifically, it performs the following tasks:
    1. Clears the Medusa attention mask in the base model.
    2. Resets the Medusa mode in the base model.
    3. Resets the current lengths in the past key-values to zero for all layers.

    Args:
    - model (MedusaLMHead): The model containing the Medusa layers and base model.
    - past_key_values (list of torch.Tensor): Contains past hidden states and past attention values.

    Returns:
    - None
    """
    model.base_model.model.mhc_mask = None
    model.base_model.model.mhc_mode = None


def reset_past_key_values(passed_key_values):
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values

def get_nucleus_one_token(logit, temperature, top_p):
    """
    Performs token sampling based on the nucleus (top-p) sampling method.

    This function selects a token from a given logit distribution using the nucleus sampling strategy.
    It allows for more controlled and diverse generation compared to traditional top-k sampling.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor (BxC).
        temperature (float): A temperature parameter to control the randomness in sampling.
                             Higher values increase diversity, lower values make selections more deterministic.
        top_p (float): The cumulative probability threshold for nucleus sampling.
                       It controls the size of the set of high-probability tokens to consider for sampling.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    if top_p >= 1:
        return torch.multinomial(F.softmax(logit / temperature, dim=-1), 1)
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_logits, dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logit[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens

def get_typical_one_token(logit, temperature, posterior_threshold, posterior_alpha):
    """
    Implements token sampling based on the typical sampling method.

    This function selects a token from a given logit distribution using the typical sampling strategy,
    aiming to balance between diversity and likelihood in a more nuanced way compared to traditional methods.

    Args:
        logit (torch.Tensor): The logits from a language model output, expected to be a 2D tensor.
        temperature (float): A parameter to control the randomness in sampling.
                              Higher values increase diversity, lower values make selections more deterministic.
        posterior_threshold (float): A threshold to decide the lower bound of probabilities to be considered for sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A tensor containing the indices of the sampled tokens.
    """
    logit = logit / temperature
    probs = torch.softmax(logit, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logit[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logit, dim=-1), 1)
    return sampled_tokens

def mhc_generate_candidates(model, base_hidden, base_token, attention_mask=None, past_key_values=None, position_ids=None, node_paths=None):
    """
    根据给定的路径参数，生成各路径的token索引
    
    Args:
        model (MHCModel): MHC模型实例
        base_hidden (torch.Tensor): 基础模型的隐藏状态
        base_token (torch.Tensor): 基础模型采样得到的token索引，形状为[batch_size, 1]
        attention_mask (torch.Tensor, optional): 注意力掩码
        past_key_values (tuple, optional): 过去的key和value状态
        position_ids (torch.Tensor, optional): 位置IDs
        node_paths (list): 路径参数，每个路径是一个整数列表，表示在每个MHC头选择的top-k索引
        
    Returns:
        padded_tokens: [路径数，最长序列长度]
        path_logits: 一个路径中logits的列表，每一个元素为[1, 序列长度, 32000]
    """

    # 存储所有中间结果的字典，键为路径元组，值为(hidden_states, logits)元组
    cache_results = {}
    # 存储每条路径生成的token索引
    path_logits = []
    path_tokens = []
    
    # 添加计数器统计mhc_forward调用次数
    # forward_count = 0
    
    # 对路径进行排序，确保依赖关系正确处理
    # sorted_paths = sorted(node_paths, key=lambda x: (len(x), x))
    
    for path in node_paths:
        # 查找最长的可复用前缀
        for prefix_len in range(len(path), 0, -1):
            prefix = tuple(path[:prefix_len])
            if prefix in cache_results:
                start_idx = prefix_len
                break
        else:
            start_idx = 0
        
        # 从最长前缀之后开始计算
        for head_idx in range(start_idx, min(len(path), model.mhc_num_heads)):
            choice_idx = path[head_idx]
            
            if head_idx == 0:
                # 第一个头直接使用base_token
                tokens_for_path = torch.tensor([], dtype=torch.long, device=base_token.device)
                logits_for_path = torch.tensor([], device=base_token.device)
                pre_hidden = base_hidden
                pre_token = base_token
            else:
                # 对于后续MHC头的结果
                prev_path = tuple(path[:head_idx])
                pre_hidden, tokens_for_path, logits_for_path = cache_results[prev_path]
                pre_token = tokens_for_path[:, -1].unsqueeze(1)
            
            # 使用mhc_forward计算当前头的输出
            # forward_count += 1  # 增加计数器
            current_hidden, current_logits = model.mhc_forward(  #current_logits [1,1,32000]
                head_idx, 
                pre_hidden,
                pre_token
            )
            k = choice_idx + 1
            token_topk = current_logits.topk(k=k, dim=-1)
            current_token = token_topk.indices[:, :, -1]       #bs,sl[1,1]

            tokens_for_path = torch.cat([tokens_for_path, current_token], dim=1)
            logits_for_path = torch.cat([logits_for_path, current_logits], dim=1)

            current_path = tuple(path[:head_idx + 1])
            cache_results[current_path] = (current_hidden, tokens_for_path, logits_for_path)
        

        path_logits.append(logits_for_path)
        path_tokens.append(tokens_for_path)
    
    # 将path_tokens转换为二维张量
    max_length = max(tokens.shape[1] for tokens in path_tokens)
    device = base_token.device
    padded_tokens = torch.zeros((len(path_tokens), max_length), dtype=torch.long, device=device)
    
    for i, tokens in enumerate(path_tokens):
        padded_tokens[i, :tokens.shape[1]] = tokens
    
    # print(f"mhc_forward 调用次数: {forward_count}")
    
    return padded_tokens, path_logits

def mhc_generate_candidates_new(model, base_hidden, base_token, attention_mask=None, past_key_values=None, position_ids=None, node_paths=None):
    """
    根据给定的路径参数，生成各路径的token索引，优化版本减少mhc_forward调用次数
    
    Args:
        model (MHCModel): MHC模型实例
        base_hidden (torch.Tensor): 基础模型的隐藏状态
        base_token (torch.Tensor): 基础模型采样得到的token索引，形状为[batch_size, 1]
        attention_mask (torch.Tensor, optional): 注意力掩码
        past_key_values (tuple, optional): 过去的key和value状态
        position_ids (torch.Tensor, optional): 位置IDs
        node_paths (list): 路径参数，每个路径是一个整数列表，表示在每个MHC头选择的top-k索引
        
    Returns:
        padded_tokens: [路径数，最长序列长度]
        path_logits: 一个路径中logits的列表，每一个元素为[1, 序列长度, 32000]
    """
    # whole_time = time.time()
    # 存储所有中间结果的字典，键为路径元组，值为(hidden_states, tokens, logits)元组
    cache_results = {}
    # 存储每条路径生成的token索引
    path_logits = []
    path_tokens = []
    
    # 添加计数器统计mhc_forward调用次数
    forward_count = 0
    
    # 初始化基础状态
    cache_results[()] = (base_hidden, torch.tensor([], dtype=torch.long, device=base_token.device), 
                         torch.tensor([], device=base_token.device))
    
    # 构建每个深度的所有唯一前缀集合
    depth_prefixes = {}
    max_depth = max(len(path) for path in node_paths)
    
    for depth in range(max_depth):
        depth_prefixes[depth] = set()
        for path in node_paths:
            if depth < len(path):
                depth_prefixes[depth].add(tuple(path[:depth]))
    
    # 为每个深度计算一次logits，然后为该深度的所有路径选择不同的token
    for depth in range(max_depth):
        if depth >= model.mhc_num_heads:
            break
            
        # 获取当前深度的所有唯一父前缀
        parent_prefixes = depth_prefixes[depth]
        
        # 对每个父前缀，计算一次mhc_forward，然后为所有子路径选择不同的token
        for parent_prefix in parent_prefixes:
            # 获取父前缀的结果
            parent_hidden, parent_tokens, parent_logits = cache_results[parent_prefix]
            
            # 获取前一个token
            if depth == 0:
                # 第一个头使用base_token
                prev_token = base_token
            else:
                # 使用父前缀生成的最后一个token
                prev_token = parent_tokens[:, -1].unsqueeze(1)
            
            # 使用mhc_forward计算当前头的输出
            current_hidden, current_logits = model.mhc_forward(
                depth, 
                parent_hidden,
                prev_token
            )
            forward_count += 1

            # 找出所有以该父前缀开头的路径，并为它们选择不同的token
            for path in node_paths:
                if depth < len(path) and tuple(path[:depth]) == parent_prefix:
                    # 获取当前路径在当前深度的选择
                    choice_idx = path[depth]
                    
                    # 根据choice_idx选择token
                    k = choice_idx + 1
                    token_topk = current_logits.topk(k=k, dim=-1)
                    current_token = token_topk.indices[:, :, -1]
                    
                    # 更新tokens和logits
                    current_tokens = torch.cat([parent_tokens, current_token], dim=1) if parent_tokens.numel() > 0 else current_token
                    current_logits_full = torch.cat([parent_logits, current_logits], dim=1) if parent_logits.numel() > 0 else current_logits
                    
                    # 缓存结果
                    current_prefix = tuple(path[:depth+1])
                    cache_results[current_prefix] = (current_hidden, current_tokens, current_logits_full)
    
    # 构建最终输出
    for path in node_paths:
        prefix = tuple(path[:min(len(path), model.mhc_num_heads)])
        if prefix in cache_results:
            _, tokens, logits = cache_results[prefix]
            path_tokens.append(tokens)
            path_logits.append(logits)
    
    # 将path_tokens转换为二维张量
    max_length = max(tokens.shape[1] for tokens in path_tokens)
    device = base_token.device
    padded_tokens = torch.zeros((len(path_tokens), max_length), dtype=torch.long, device=device)
    
    for i, tokens in enumerate(path_tokens):
        padded_tokens[i, :tokens.shape[1]] = tokens

    # whole_forward_time = time.time() - whole_time forward_count
    # print(f"whole_forward_time: {whole_forward_time:.4f}s ")
    print(f"forward_count: {forward_count}")
    return padded_tokens, path_logits


def generate_candidates(model, base_hidden, base_logits, retrieve_indices, flatten_len, node_paths=None, temperature=0, posterior_threshold=0.3, posterior_alpha=0.09, top_p=0.8, sampling='typical', fast=False):
    """
    根据base_hidden和node_paths生成各路径的token索引，并将其展平
    
    Parameters:
    - model (MHCModel): MHC模型实例
    - base_hidden (torch.Tensor): 基础模型的隐藏状态
    - base_logits (torch.Tensor): 基础模型的输出logits
    - node_paths (list): 路径参数，每个路径是一个整数列表
    - temperature (float): 温度参数
    - posterior_threshold (float): 后验阈值
    - posterior_alpha (float): 后验alpha参数
    - top_p (float): top-p采样参数
    - sampling (str): 采样方法，'typical'或'nucleus'
    - fast (bool): 是否使用快速模式
    
    Returns:
    - tuple: 包含candidates、flatten_candidates、mhc_attn_mask和mhc_position_ids
    """
    device = base_hidden.device
    
    # 对node_paths进行排序
    # sorted_paths = sorted(node_paths, key=lambda x: (len(x), x))
    
    # 从base_logits采样得到base_token
    if temperature == 0 or fast:
        base_token = torch.argmax(base_logits[:, -1:], dim=-1)
    else:
        if sampling == 'typical':
            base_token = get_typical_one_token(base_logits[:, -1], temperature, posterior_threshold, posterior_alpha)
        elif sampling == 'nucleus':
            base_token = get_nucleus_one_token(base_logits[:, -1], temperature, top_p)
        else:
            raise NotImplementedError
    
    # 使用mhc_generate_candidates生成各路径的token索引
    path_tokens, path_logits = mhc_generate_candidates_new(
        model=model,
        base_hidden=base_hidden,
        base_token=base_token,
        node_paths=node_paths
    )
    
    # 将path_tokens转换为candidates（添加base_token列）
    candidates = torch.zeros((path_tokens.shape[0], path_tokens.shape[1] + 1), dtype=torch.long, device=device)
    candidates[:, 0] = base_token.item()
    candidates[:, 1:] = path_tokens
    
    # 创建展平的候选token列表
    flatten_candidates = torch.zeros(flatten_len, dtype=torch.long, device=device)
    
    # 填充base token
    flatten_candidates[0] = base_token.item()
    
    # 使用retrieve_indices直接填充其他token
    for path_idx in range(retrieve_indices.shape[0]):
        for depth in range(1, retrieve_indices.shape[1]):
            flatten_idx = retrieve_indices[path_idx, depth]
            if flatten_idx >= 0:  # 确保有效的索引
                token_idx = candidates[path_idx, depth]
                flatten_candidates[flatten_idx] = token_idx
    
    flatten_candidates = flatten_candidates.unsqueeze(0)
    return candidates, flatten_candidates, path_logits


def tree_decoding(
    model,
    tree_candidates,
    past_key_values,
    mhc_position_ids,
    input_ids,
    retrieve_indices,
):
    """
    Decode the tree candidates using the provided model and reorganize the logits.
    
    Parameters:
    - model (nn.Module): Model to be used for decoding the tree candidates.
    - tree_candidates (torch.Tensor): Input candidates based on a tree structure.
    - past_key_values (torch.Tensor): Past states, such as key and value pairs, used in attention layers.
    - medusa_position_ids (torch.Tensor): Positional IDs associated with the Medusa structure.
    - input_ids (torch.Tensor): Input sequence IDs.
    - retrieve_indices (list or torch.Tensor): Indices for reordering the logits.
    
    Returns:
    - tuple: Returns medusa logits, regular logits, and other outputs from the model.
    """

    # Compute new position IDs by adding the Medusa position IDs to the length of the input sequence.
    position_ids = mhc_position_ids + input_ids.shape[1]

    # Use the model to decode the tree candidates. 
    # The model is expected to return logits for the Medusa structure, original logits, and possibly other outputs.
    _, outputs, tree_logits = model(
        tree_candidates,
        past_key_values=past_key_values,
        position_ids=position_ids,
        inference_mode=True,
    )
    
    # Reorder the obtained logits based on the retrieve_indices to ensure consistency with some reference ordering.
    logits = tree_logits[0, retrieve_indices]
    return outputs, logits

def get_nucleus_posterior_mask(logits, candidates, temperature, top_p):
    """
    Generates a posterior mask for token candidates using nucleus (top-p) sampling.

    This function applies nucleus sampling to a set of logits, and then generates a mask indicating 
    which candidate tokens are selected. It adapts the sampling strategy to accommodate for 
    temperature scaling and cumulative probability thresholding.

    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        top_p (float): The cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    # adapted from https://github.com/huggingface/transformers/blob/18a879f47576822aa1a5c49aecb27d89bfa5fa69/examples/run_generation.py#L79

    # Apply temperature
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    if top_p >= 1:
        sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
        sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
        posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
        return posterior_mask
    # Convert to probabilities (softmax)
    probs = F.softmax(logits, dim=-1)
    # Sort the probabilities
    sorted_logits, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cum_probs = torch.cumsum(sorted_logits, dim=-1)

    # Create mask for the top-p nucleus
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

    
    # Remove low-probability tokens
    logits[indices_to_remove] = float('-inf')
    # Sample from the remaining tokens
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    # Create a mask for selected tokens
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()

    return posterior_mask

def get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha):
    """
    Args:
        logits (torch.Tensor): A tensor of logits from a language model output.
        candidates (torch.Tensor): A tensor of candidate tokens to compare against sampled tokens.
        temperature (float): A parameter to scale the logits, controlling randomness in sampling.
        posterior_threshold (float): The minimum threshold for probabilities to be considered in sampling.
        posterior_alpha (float): A scaling factor applied to the entropy-based adaptive threshold.

    Returns:
        torch.Tensor: A posterior mask indicating which candidate tokens match the sampled tokens.
    """
    logits = logits[:, :-1] / temperature
    n_samples, n_tokens = logits.shape[0], logits.shape[1]
    logits = logits.view(n_samples*n_tokens, -1)
    probs = F.softmax(logits, dim=-1)
    entropy = -torch.sum(
            probs * torch.log(probs + 1e-5), dim=-1
        )
    threshold = torch.minimum(
            torch.ones_like(entropy) * posterior_threshold,
            torch.exp(-entropy) * posterior_alpha,
        )
    indices_to_remove = probs < threshold.unsqueeze(-1)
    logits[indices_to_remove] = float('-inf')
    sampled_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)
    sampled_tokens = sampled_tokens.view(n_samples, n_tokens)
    posterior_mask = (candidates[:, 1:] == sampled_tokens).int()
    return posterior_mask

def evaluate_posterior(
    logits, candidates, temperature, posterior_threshold=0.3, posterior_alpha = 0.09, top_p=0.8, sampling = 'typical', fast = True
):
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.
    - top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    - sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    - fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if temperature == 0:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
            candidates[:, 1:] == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
        
    if sampling == 'typical':
        if fast:
            posterior_prob = torch.softmax(logits[:, :-1] / temperature, dim=-1)
            candidates_prob = torch.gather(
                posterior_prob, dim=-1, index=candidates[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            posterior_entropy = -torch.sum(
                posterior_prob * torch.log(posterior_prob + 1e-5), dim=-1
            )  # torch.sum(torch.log(*)) is faster than torch.prod
            threshold = torch.minimum(
                torch.ones_like(posterior_entropy) * posterior_threshold,
                torch.exp(-posterior_entropy) * posterior_alpha,
            )
            posterior_mask = candidates_prob > threshold
            candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)

            # Choose the best candidate based on the evaluated posterior probabilities
            accept_length = candidates_accept_length.max()
            if accept_length == 0:
                # If no candidates are accepted, just choose the first one
                best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
            else:
                best_candidates = torch.where(candidates_accept_length == accept_length)[0]
                # Accept the best one according to likelihood
                likelihood = torch.sum(
                    torch.log(candidates_prob[best_candidates, :accept_length]), dim=-1
                )
                best_candidate = best_candidates[torch.argmax(likelihood)]
            return best_candidate, accept_length
        # Calculate posterior probabilities and thresholds for candidate selection
        posterior_mask = get_typical_posterior_mask(logits, candidates, temperature, posterior_threshold, posterior_alpha, fast)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        # Choose the best candidate based on the evaluated posterior probabilities
        accept_length = candidates_accept_length.max()
        
        if accept_length == 0:
            # If no candidates are accepted, just choose the first one
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
            # Accept the best one according to likelihood
        return best_candidate, accept_length
    
    if sampling == 'nucleus':
        assert top_p < 1.0 + 1e-6, "top_p should between 0 and 1"
        posterior_mask = get_nucleus_posterior_mask(logits, candidates, temperature, top_p)
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length
    else:
        raise NotImplementedError

def update_inference_inputs(
    input_ids,
    candidates,
    best_candidate,
    accept_length,
    retrieve_indices,
    outputs,
    logits,
    new_token,
    past_key_values_data,
    current_length_data,
):
    """
    Update the input sequences and relevant tensors based on the selected best candidate from the inference results.

    Args:
    - input_ids (torch.Tensor): Current input token sequences.
    - candidates (torch.Tensor): Candidate token sequences generated in the current step.
    - best_candidate (int): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    - retrieve_indices (torch.Tensor): Indices to map tree to a cartesian product.
    - outputs, logits (torch.Tensor): Model's outputs from the previous inference step.
    - new_token (int): Counter for the new tokens added during inference.
    - past_key_values_data (torch.Tensor): Tensor containing past hidden states for the transformer model.
    - current_length_data (torch.Tensor): Tensor containing the current length of sequences in the batch.

    Returns:
    - input_ids (torch.Tensor): Updated input token sequences.
    - logits (torch.Tensor): Updated logits.
    - new_token (int): Updated counter for the new tokens added.
    - base_hidden (torch.Tensor): Hidden state of the base model.
    """
    # Calculate the starting position for new tokens based on the previous input length
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
        retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1]], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    tgt = past_key_values_data[..., select_indices, :]
    # Destination tensor where the relevant past information will be stored
    dst = past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
    # Copy relevant past information from the source to the destination
    dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    # Extract logits and medusa logits for the accepted tokens
    logits = logits[best_candidate, accept_length : accept_length + 1, :].unsqueeze(1)
    # logits = logits[None, best_candidate, accept_length : accept_length + 1]
    # Get the base hidden states for the next iteration
    base_hidden = outputs[0][..., retrieve_indices[best_candidate, accept_length], :].unsqueeze(1)
    # Update the new token counter
    new_token += accept_length + 1

    return input_ids, logits, new_token, base_hidden
