import warnings

warnings.filterwarnings("ignore")

import os
import sys
from typing import List
import numpy as np
import fire
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, AutoConfig
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from dataclasses import dataclass
import torch.nn as nn
import math
import warnings
from functools import partial
import numpy as np
import fire
import transformers
from torch.optim.lr_scheduler import LambdaLR
import json
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import (
    D3Dataset,
    SFTHistoryTitle2TargetTitleDataset,
    SFTHistorySid2TargetSidDataset,
    SFTSidxTilteDataset,
    SFTHistorySid2FeatDataset,
    PreferenceSFTDataset,
    UserPreference2sidSFTDataset,
    TitleHistory2SidSFTDataset,
)
import random
from datasets import Dataset as HFDataset
from torch.utils.data import ConcatDataset
from rich_logger import logger


class TokenExtender:
    def __init__(self, data_path):
        self.data_path = data_path
        self.indices = None
        self.new_tokens = None

    def _load_data(self):
        with open(self.data_path, "r") as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens

        if self.indices is None:
            self._load_data()

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(
        0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def process_train_dataset(train_datasets, seed=42):
    """
    处理训练数据集，保持 CL_prefixes 顺序（2 > 1 > 0），组内随机重排，
    并将其他数据集的样本随机插入到当前数据集中。

    Args:
        train_datasets: 原始数据集列表
        seed: 随机种子

    Returns:
        HFDataset: 处理后的训练数据集
    """
    # 1. 分离 CL_prefixes 数据集和其他数据集
    cl_datasets = []
    other_datasets = []

    # 前 6 个是 CL_prefixes 数据集（2个任务 × 3种前缀）
    cl_datasets.extend(train_datasets[:6])
    # 剩余的是其他数据集
    other_datasets.extend(train_datasets[6:])

    # 2. 分组洗牌 CL_prefixes 数据集（保持 CL_prefixes 顺序 2 > 1 > 0，组内随机）
    shuffled_cl_datasets = []

    # CL_prefixes=2 的数据集（索引 0, 3）
    for i in range(0, 6, 3):
        temp_dataset = HFDataset.from_dict(
            {k: [v[k] for v in cl_datasets[i]] for k in cl_datasets[i][0].keys()}
        )
        temp_dataset = temp_dataset.shuffle(seed=seed)
        shuffled_cl_datasets.append(temp_dataset)

    # CL_prefixes=1 的数据集（索引 1, 4）
    for i in range(1, 6, 3):
        temp_dataset = HFDataset.from_dict(
            {k: [v[k] for v in cl_datasets[i]] for k in cl_datasets[i][0].keys()}
        )
        temp_dataset = temp_dataset.shuffle(seed=seed)
        shuffled_cl_datasets.append(temp_dataset)

    # CL_prefixes=0 的数据集（索引 2, 5）
    for i in range(2, 6, 3):
        temp_dataset = HFDataset.from_dict(
            {k: [v[k] for v in cl_datasets[i]] for k in cl_datasets[i][0].keys()}
        )
        temp_dataset = temp_dataset.shuffle(seed=seed)
        shuffled_cl_datasets.append(temp_dataset)

    # 合并 CL_prefixes 数据集
    cl_combined = concatenate_datasets(shuffled_cl_datasets)

    # 3. 处理其他数据集，转换为 HFDataset 并洗牌
    shuffled_other_datasets = []
    for dataset in other_datasets:
        temp_dataset = HFDataset.from_dict(
            {k: [v[k] for v in dataset] for k in dataset[0].keys()}
        )
        temp_dataset = temp_dataset.shuffle(seed=seed)
        shuffled_other_datasets.append(temp_dataset)

    # 合并其他数据集
    other_combined = (
        concatenate_datasets(shuffled_other_datasets)
        if shuffled_other_datasets
        else None
    )

    # 4. 将其他数据集的样本随机插入到 CL_prefixes 数据集中
    if other_combined:
        # 获取两个数据集的索引
        cl_indices = list(range(len(cl_combined)))
        other_indices = list(range(len(other_combined)))

        # 随机化插入位置
        np.random.seed(seed)
        np.random.shuffle(other_indices)

        # 创建新的数据集列表
        combined_indices = []
        cl_ptr = 0
        other_ptr = 0

        while cl_ptr < len(cl_indices) and other_ptr < len(other_indices):
            # 随机决定是否插入其他数据集的样本
            if np.random.random() < 0.5:  # 50% 概率插入
                combined_indices.append((1, other_indices[other_ptr]))
                other_ptr += 1
            else:
                combined_indices.append((0, cl_indices[cl_ptr]))
                cl_ptr += 1

        # 添加剩余的样本
        while cl_ptr < len(cl_indices):
            combined_indices.append((0, cl_indices[cl_ptr]))
            cl_ptr += 1

        while other_ptr < len(other_indices):
            combined_indices.append((1, other_indices[other_ptr]))
            other_ptr += 1

        # 构建最终的数据集
        final_data = {}
        for key in cl_combined.features:
            final_data[key] = []

        for dtype, idx in combined_indices:
            if dtype == 0:
                # 来自 CL_prefixes 数据集
                for key in cl_combined.features:
                    final_data[key].append(cl_combined[idx][key])
            else:
                # 来自其他数据集
                for key in cl_combined.features:
                    final_data[key].append(other_combined[idx][key])

        hf_train_dataset = HFDataset.from_dict(final_data)
    else:
        hf_train_dataset = cl_combined

    return hf_train_dataset


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    freeze_LLM: bool = False,  # freeze LLM parameters, only train new token embeddings
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    category: str = "",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
):
    set_seed(seed)
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
        "All_Beauty": "all beauty",
    }
    print(category)
    category = category_dict[category]
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
        )
    else:
        config = AutoConfig.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_config(config)
        print("Training from scratch!")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    original_vocab_size = model.get_input_embeddings().weight.shape[0]

    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(data_path=sid_index_path)
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    # Freeze LLM parameters if required
    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if sid_index_path and os.path.exists(sid_index_path) and new_tokens:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    # grad shape: [vocab_size, hidden_dim]
                    grad[:original_vocab_size].zero_()
                    return grad

                embedding_layer.weight.register_hook(mask_grad)

                print(
                    f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})"
                )

        else:
            print(
                "Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!"
            )

        # Print the number of trainable parameters (it will still report the size of the entire embedding matrix, but only the newly added rows will have non-zero gradients).
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100*trainable_params/total_params:.2f}%)"
        )

    train_datasets = []
    # * [课程学习] history_sids -> target_sid
    data_historySID_to_targetSID_CL_prefixes_2 = SFTHistorySid2TargetSidDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        CL_prefixes=2,
    )

    data_historySID_to_targetSID_CL_prefixes_1 = SFTHistorySid2TargetSidDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        CL_prefixes=1,
    )

    data_historySID_to_targetSID_CL_prefixes_0 = SFTHistorySid2TargetSidDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        CL_prefixes=0,
    )

    train_datasets.append(data_historySID_to_targetSID_CL_prefixes_2)
    train_datasets.append(data_historySID_to_targetSID_CL_prefixes_1)
    train_datasets.append(data_historySID_to_targetSID_CL_prefixes_0)

    # * sid -> title, title -> sid
    data_sidxtitle_CL_prefixes_2 = SFTSidxTilteDataset(
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        CL_prefixes=2,
    )

    data_sidxtitle_CL_prefixes_1 = SFTSidxTilteDataset(
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        CL_prefixes=1,
    )

    data_sidxtitle_CL_prefixes_0 = SFTSidxTilteDataset(
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        CL_prefixes=0,
    )

    train_datasets.append(data_sidxtitle_CL_prefixes_2)
    train_datasets.append(data_sidxtitle_CL_prefixes_1)
    train_datasets.append(data_sidxtitle_CL_prefixes_0)

    # * history_sids -> title / description
    data_historySID_to_feat = SFTHistorySid2FeatDataset(
        train_file=train_file,
        item_file=item_meta_path,
        index_file=sid_index_path,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
    )

    train_datasets.append(data_historySID_to_feat)

    # * history_titles -> target_title
    data_historyTitle_to_targetTitle = SFTHistoryTitle2TargetTitleDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
    )
    train_datasets.append(data_historyTitle_to_targetTitle)
    # 调用函数处理训练数据集
    hf_train_dataset = process_train_dataset(train_datasets, seed=42)

    # * history_sids -> target_sid
    val_data = SFTHistorySid2TargetSidDataset(
        train_file=eval_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
    )
    # val_data = SFTHistoryTitle2TargetTitleDataset(train_file=eval_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=20000, seed=seed, category=category)
    print("LOAD DATA FINISHED")

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    model.gradient_checkpointing_enable()

    sample_frac = 1
    hf_train_dataset = hf_train_dataset.select(
        range(int(sample_frac * len(hf_train_dataset)))
    )
    hf_val_dataset = HFDataset.from_dict(
        {k: [v[k] for v in val_data] for k in val_data[0].keys()}
    )

    print(hf_train_dataset)
    print(hf_val_dataset)
    eval_step = 100
    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=eval_step,
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="none",
            gradient_checkpointing=True,  # 启用梯度检查点
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        # optimizers=(optimizer, lr_scheduler)
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
