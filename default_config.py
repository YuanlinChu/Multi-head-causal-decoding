MISTRAL_CONFIG = {
    # 模型参数
    "model_args": {
        # "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
        "model_name_or_path": "/hpc2hdd/home/ychu763/Documents/Medusa/mistralai/Mistral-7B-Instruct-v0.2",
    },
    
    # 数据参数
    "data_args": {
        "data_path": "data/mistral.json",
        "lazy_preprocess": True,
    },
    
    # 训练参数
    "training_args": {
        # Medusa 特定参数
        "mhc_num_heads": 5,
        "res_layer_nums": 1,
        
        # 基础训练参数
        "model_max_length": 2048,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 2,
        
        # 优化器设置
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        
        # 输出和评估
        "output_dir": "test",
        "eval_strategy": "no",
        # "save_strategy": "no",
        "save_strategy": "steps",  # 改为"steps"
        "save_steps": 200,         # 每200步保存一次
        "logging_steps": 1,
        
        # 精度设置
        "bf16": True,
        "tf32": True,
        
        # 分布式训练
        "ddp_find_unused_parameters": False,

        # DeepSpeed 配置
        # "deepspeed": "deepspeed.json",
    }
} 

VICUNA_CONFIG = {
    # 模型参数
    "model_args": {
        "model_name_or_path": "lmsys/vicuna-7b-v1.3",
    },
    
    # 数据参数
    "data_args": {
        "data_path": "data/ShareGPT.json",
        "lazy_preprocess": True,
    },
    
    # 训练参数
    "training_args": {
        # Medusa 特定参数
        "mhc_num_heads": 5,
        "res_layer_nums": 1,
        
        # 基础训练参数
        "model_max_length": 2048,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 4,
        
        # 优化器设置
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        
        # 输出和评估
        "output_dir": "shareGPT-4epochs",
        "eval_strategy": "no",
        # "save_strategy": "no",
        "save_strategy": "steps",  # 改为"steps"
        "save_steps": 1000,         # 每200步保存一次
        "logging_steps": 1,
        
        # 精度设置
        "bf16": True,
        "tf32": True,
        
        # 分布式训练
        "ddp_find_unused_parameters": False,

        # DeepSpeed 配置
        "deepspeed": "deepspeed.json",
    }
} 