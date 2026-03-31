import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
from scipy import sparse as sp
from tqdm import tqdm
from torch.nn import functional as F
from rich_logger import logger
import argparse
import matplotlib.pyplot as plt
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
import heapq
import os
import sys
from collections import defaultdict

sys.path.append(
    "/ssd/hust-tangxi/workspace/MOR/src/residual_quantization/CF/LightGCN-PyTorch/code"
)

from models.rqvae import RQVAE
from models.layers import MLPLayers

original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
from model import LightGCN

sys.argv = original_argv


class JointCFRQVAE(nn.Module):
    def __init__(self, config, dataset, sid_type="cf"):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.sid_type = sid_type

        self.rqvae = RQVAE(
            in_dim=dataset.emb_dim,
            num_emb_list=config["rqvae"]["num_emb_list"],
            e_dim=config["rqvae"]["e_dim"],
            layers=config["rqvae"]["layers"],
            dropout_prob=config["rqvae"]["dropout_prob"],
            bn=config["rqvae"]["bn"],
            loss_type=config["rqvae"]["loss_type"],
            quant_loss_weight=config["rqvae"]["quant_loss_weight"],
            beta=config["rqvae"]["beta"],
            kmeans_init=config["rqvae"]["kmeans_init"],
            kmeans_iters=config["rqvae"]["kmeans_iters"],
            sk_epsilons=config["rqvae"]["sk_epsilons"],
            sk_iters=config["rqvae"]["sk_iters"],
        )

        self.lightgcn = None
        if self.sid_type == "cf":
            self.lightgcn = LightGCN(
                config=config["lightgcn"],
                dataset=dataset,
            )

    def forward(self, users, pos_items, neg_items, use_sk=True):
        multimodal_content_item_emb = torch.FloatTensor(self.dataset.embeddings).cuda()

        rqvae_rec_emb, last_quantized_emb, quant_loss, indices = self.rqvae(
            multimodal_content_item_emb, use_sk=use_sk
        )

        loss_recon = F.mse_loss(
            rqvae_rec_emb, multimodal_content_item_emb, reduction="mean"
        )

        cf_bpr_loss = torch.tensor(0.0, device=multimodal_content_item_emb.device)
        cf_reg_loss = torch.tensor(0.0, device=multimodal_content_item_emb.device)
        align_loss = torch.tensor(0.0, device=multimodal_content_item_emb.device)

        if self.sid_type == "cf" and self.lightgcn is not None:
            cf_item_emb = self.lightgcn.embedding_item.weight

            pos_last_quantized_emb = last_quantized_emb[pos_items]
            pos_cf_item_emb = cf_item_emb[pos_items]
            align_loss = F.l1_loss(
                pos_cf_item_emb, pos_last_quantized_emb, reduction="mean"
            )

            cf_bpr_loss, cf_reg_loss = self.lightgcn.bpr_loss(
                users, pos_items, neg_items
            )

        lambda_align = 0.5
        if self.sid_type == "cf":
            total_loss = (
                loss_recon
                + self.config["rqvae"]["quant_loss_weight"] * quant_loss
                + cf_bpr_loss
                + cf_reg_loss
                + lambda_align * align_loss
            )
        else:
            total_loss = (
                loss_recon + self.config["rqvae"]["quant_loss_weight"] * quant_loss
            )

        loss_dict = {
            "rq_loss_recon_loss": loss_recon.item(),
            "rq_quant_loss": quant_loss.item(),
            "cf_bpr_loss": cf_bpr_loss.item(),
            "cf_reg_loss": cf_reg_loss.item(),
            "align_loss": align_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, loss_dict

    def get_users_rating(self, users):
        if self.sid_type == "cf" and self.lightgcn is not None:
            return self.lightgcn.getUsersRating(users)
        else:
            num_items = self.dataset.m_items
            return torch.rand(len(users), num_items, device=users.device)
