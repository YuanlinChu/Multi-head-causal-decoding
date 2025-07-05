# MHC: Multi-Head-Casual Language Model

## 简介

MHC（Multi-Head-Casual）是一个基于多解码头的语言模型加速框架，灵感来源于Medusa项目。与传统的多头解码方法不同，MHC在连续的解码头之间引入了因果机制，实现了更高效的并行推理。

### 核心特性

- **因果多头解码**：每个解码头依赖于前一个解码头的隐藏层输出和输出token的embedding向量
- **级联依赖设计**：通过concat操作将前一头的输出与当前头的输入结合，经过维度变化和残差层处理
- **高效推理**：通过树形注意力机制和候选验证，实现比传统自回归解码更快的推理速度
- **兼容性强**：支持Vicuna、Mistral等主流语言模型

### 架构特点

```
输入 → 基础模型 → 隐藏状态
                      ↓
Head1: 隐藏状态 → ResBlock → 输出1
                      ↓
Head2: [隐藏状态 + 输出1嵌入] → ResBlock → 输出2
                      ↓
Head3: [隐藏状态 + 输出2嵌入] → ResBlock → 输出3
                      ↓
                     ...
```

每个MHC头通过以下方式处理：
1. 将前一头的隐藏状态和输出token embedding进行concat
2. 通过维度变换层调整到合适的hidden size
3. 经过多个ResBlock残差层处理
4. 输出当前位置的预测token

## 安装

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- transformers >= 4.28.0

### 安装步骤

1. 克隆项目：
```bash
git clone <repository_url>
cd my_model
```

2. 安装依赖：
```bash
pip install torch transformers accelerate deepspeed
pip install fastchat wandb tqdm shortuuid
pip install safetensors huggingface_hub
```

3. 安装额外依赖（可选）：
```bash
pip install bitsandbytes  # 用于4bit/8bit量化
```

## 训练

### 数据准备

1. 准备训练数据（ShareGPT格式）：
```bash
# 下载ShareGPT数据集
git clone https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered
```

2. 数据预处理：
```bash
python data/create_data.py --input-filename ShareGPT_Vicuna_unfiltered/ShareGPT_V4.3_unfiltered_cleaned_split.json --output-filename data/ShareGPT.json
```

### 单GPU训练

```bash
python train.py \
    --model_name_or_path lmsys/vicuna-7b-v1.3 \
    --data_path data/ShareGPT.json \
    --output_dir ./output_mhc \
    --mhc_num_heads 5 \
    --res_layer_nums 1 \
    --bf16 True \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --save_steps 1000 \
    --model_max_length 2048 \
    --lazy_preprocess True
```

### 多GPU训练（推荐）

使用DeepSpeed进行多GPU训练：

```bash
torchrun --nproc_per_node=4 train.py \
    --model_name_or_path lmsys/vicuna-7b-v1.3 \
    --data_path data/ShareGPT.json \
    --output_dir ./output_mhc \
    --mhc_num_heads 5 \
    --res_layer_nums 1 \
    --bf16 True \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --save_steps 1000 \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --deepspeed deepspeed.json
```

### 训练参数说明

- `--mhc_num_heads`: MHC头的数量，默认5
- `--res_layer_nums`: 每个MHC头的残差层数量，默认1
- `--learning_rate`: 学习率，建议1e-3（因为只训练新增的头部）
- `--deepspeed`: DeepSpeed配置文件路径

## 推理

### 命令行对话

启动交互式命令行界面：

```bash
python cli.py --model <model_path>
```

示例：
```bash
python cli.py --model ./output_mhc/checkpoint-2000
```

### 推理参数

```bash
python cli.py \
    --model ./output_mhc/checkpoint-2000 \
    --temperature 0.0 \
    --max_steps 512 \
    --load-in-8bit  # 可选：8bit量化
```

### 命令行交互命令

在对话界面中可以使用以下命令：
- `!!exit` 或空行：退出程序
- `!!reset`：重置对话
- `!!remove`：删除最后一条消息
- `!!regen`：重新生成最后一条回复
- `!!save <filename>`：保存对话历史
- `!!load <filename>`：加载对话历史

## 评估

### MT-Bench评估

1. 生成模型回答：
```bash
CUDA_VISIBLE_DEVICES=0 python eval/gen_mhc_model_answer.py \
    --model-path ./output_mhc/checkpoint-2000 \
    --model-id mhc-vicuna-7b-v1.3 \
    --temperature 0.0 \
    --posterior_threshold 0.09 \
    --posterior_alpha 0.3
```

2. 生成baseline回答：
```bash
CUDA_VISIBLE_DEVICES=0 python eval/gen_mhc_model_answer_baseline.py \
    --model-path ./output_mhc/checkpoint-2000 \
    --model-id mhc-vicuna-7b-v1.3-baseline
```

### 速度测试

```bash
CUDA_VISIBLE_DEVICES=0 python eval/speed.py \
    --model-path ./output_mhc/checkpoint-2000 \
    --mhc_choices mc_sim_7b_63
```

## 配置文件

项目使用`default_config.py`管理训练配置，支持：

- **VICUNA_CONFIG**: Vicuna模型配置
- **MISTRAL_CONFIG**: Mistral模型配置

可以根据需要修改配置文件中的参数。

## 项目结构

```
my_model/
├── train.py              # 训练脚本
├── cli.py                # 命令行推理接口
├── model.py              # MHC模型定义
├── utils.py              # 工具函数
├── default_config.py     # 默认配置
├── mhc_choices.py        # MHC选择策略
├── data/                 # 数据处理脚本
├── eval/                 # 评估脚本
│   ├── gen_mhc_model_answer.py
│   ├── gen_mhc_model_answer_baseline.py
│   └── speed.py
└── deepspeed.json        # DeepSpeed配置
```

## 性能特点

- **推理加速**：通过多头并行预测实现约2x的推理加速
- **内存效率**：只需训练新增的MHC头部，显存需求相对较小
- **模型兼容**：支持主流的Llama、Vicuna、Mistral等模型架构

## 注意事项

1. 训练时只训练新增的MHC头部，基础模型参数保持冻结
2. 推荐使用DeepSpeed进行多GPU训练以提高效率
3. 推理时支持4bit/8bit量化以减少显存使用
4. 评估时可以调整`posterior_threshold`和`posterior_alpha`参数优化性能

## 致谢

本项目基于以下优秀的开源项目：
- [Medusa](https://github.com/FasterDecoding/Medusa)
- [FastChat](https://github.com/lm-sys/FastChat)
- [Transformers](https://github.com/huggingface/transformers)
