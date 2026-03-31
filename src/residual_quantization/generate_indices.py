import collections
import json
import logging
import argparse
import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import EmbDataset, JointEmbInterDataset
from models.rqvae import RQVAE
from models.joint_cf_rqvae import JointCFRQVAE
import os
from rich_logger import logger


def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for inference"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    ckpt = torch.load(
        args.ckpt_path, map_location=torch.device("cpu"), weights_only=False
    )

    ckpt_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    config = {
        "rqvae": {
            "num_emb_list": ckpt_args.num_emb_list,
            "e_dim": ckpt_args.e_dim,
            "layers": ckpt_args.rqvae_layers,
            "dropout_prob": ckpt_args.dropout_prob,
            "bn": ckpt_args.bn,
            "loss_type": ckpt_args.loss_type,
            "quant_loss_weight": ckpt_args.quant_loss_weight,
            "beta": ckpt_args.beta,
            "kmeans_init": ckpt_args.kmeans_init,
            "kmeans_iters": ckpt_args.kmeans_iters,
            "sk_epsilons": ckpt_args.sk_epsilons,
            "sk_iters": ckpt_args.sk_iters,
        },
        "lightgcn": {
            "latent_dim_rec": ckpt_args.latent_dim_rec,
            "lightGCN_n_layers": ckpt_args.lightGCN_n_layers,
            "keep_prob": ckpt_args.keep_prob,
            "A_split": ckpt_args.A_split,
            "pretrain": ckpt_args.pretrain,
            "dropout": ckpt_args.lgcn_dropout,
        },
        "train": {
            "epochs": ckpt_args.epochs,
            "batch_size": ckpt_args.batch_size,
            "lr": ckpt_args.lr,
            "weight_decay": ckpt_args.weight_decay,
            "early_stop_patience": ckpt_args.early_stop_patience,
        },
        "test": {
            "batch_size": ckpt_args.batch_size,
        },
    }

    joint_emb_inter_dataset = JointEmbInterDataset(
        interaction_path=ckpt_args.interaction_path,
        emb_path=ckpt_args.emb_path,
        test_ratio=ckpt_args.test_ratio,
        seed=ckpt_args.seed,
        split=ckpt_args.A_split,
        folds=100,
    )

    model = JointCFRQVAE(config, joint_emb_inter_dataset, sid_type=ckpt_args.sid_type)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    data = EmbDataset(ckpt_args.emb_path)
    data_loader = DataLoader(
        data,
        num_workers=10,
        batch_size=ckpt_args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in tqdm(data_loader, ncols=100, desc=f"Generating Indices"):
        d = d.to(device)
        indices = model.rqvae.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    for vq in model.rqvae.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0

    if model.rqvae.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rqvae.rq.vq_layers[-1].sk_epsilon = 0.003

    tt = 0
    while True:
        if tt >= 20 or check_collision(all_indices_str):
            break

        collision_item_groups = get_collision_item(all_indices_str)
        print(collision_item_groups)
        print(len(collision_item_groups))
        for collision_items in collision_item_groups:
            d = data[collision_items].to(device)

            indices = model.rqvae.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                code = []
                for i, ind in enumerate(index):
                    code.append(prefix[i].format(int(ind)))

                all_indices[item] = code
                all_indices_str[item] = str(code)
        tt += 1

    print("All indices number: ", len(all_indices))
    print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate", (tot_item - tot_indice) / tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    with open(args.output_path, "w") as fp:
        json.dump(all_indices_dict, fp)

    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
