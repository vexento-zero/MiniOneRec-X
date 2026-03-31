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
from models.joint_cf_rqvae import JointCFRQVAE
from datasets import JointEmbInterDataset


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)

    precision = RecallPrecision_ATk(groundTrue, r, 20)["precision"]
    recall = RecallPrecision_ATk(groundTrue, r, 20)["recall"]
    ndcg = NDCGatK_r(groundTrue, r, 20)

    return precision, recall, ndcg


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype("float")


def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {"recall": recall, "precision": precis}


def NDCGatK_r(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1

    ndcg = 0
    for i in range(len(pred_data)):
        dcg_max = np.sum(test_matrix[i] / np.log2(np.arange(2, k + 2)))
        dcg_pred = np.sum(pred_data[i] / np.log2(np.arange(2, k + 2)))
        if dcg_max == 0:
            continue
        ndcg += dcg_pred / dcg_max
    return ndcg


def Test(dataset, model, config):
    batch_size = config["test"]["batch_size"]
    testDict = dataset.testDict
    model = model.eval()
    max_K = 20

    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []

        total_batch = len(users) // batch_size + 1
        for batch_users in minibatch(users, batch_size=batch_size):
            allPos = dataset.allPos
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long().cuda()
            rating = model.get_users_rating(batch_users_gpu)

            exclude_index = []
            exclude_items = []
            for uid, items in enumerate(allPos):
                exclude_index.extend([uid] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        X = zip(rating_list, groundTrue_list)

        precision = []
        recall = []
        ndcg = []

        for x in X:
            p, r, n = test_one_batch(x)
            precision.append(p)
            recall.append(r)
            ndcg.append(n)

        precision = np.array(precision).mean()
        recall = np.array(recall).mean()
        ndcg = np.array(ndcg).mean()

        collision_rate = calculate_collision_rate(model, dataset.embeddings)

        return precision, recall, ndcg, collision_rate


def calculate_collision_rate(model, data):
    model.eval()
    indices_set = set()
    num_sample = len(data)
    data = torch.FloatTensor(data).cuda()
    indices = model.rqvae.get_indices(data)
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = "-".join([str(int(_)) for _ in index])
        indices_set.add(code)

    collision_rate = (num_sample - len(list(indices_set))) / num_sample

    return collision_rate


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get("batch_size", 1024)
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i : i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i : i + batch_size] for x in tensors)


def plot_loss_curves(loss_history, sid_type, output_dir):
    base = ["total_loss", "rq_loss_recon_loss", "rq_quant_loss", "val_collision_rate"]
    extra = [
        "cf_bpr_loss",
        "cf_reg_loss",
        "align_loss",
        "val_precision",
        "val_recall",
        "val_ndcg",
    ]

    loss_types = base + (extra if sid_type == "cf" else [])
    n_plots = len(loss_types)

    fig, axes = plt.subplots(
        (n_plots + 2) // 3,
        3,
        figsize=(18, 4 * ((n_plots + 2) // 3)),
        facecolor="#f8f9fa",
    )

    colors = plt.cm.Set3(np.linspace(0, 1, n_plots))

    for idx, (ax, key) in enumerate(zip(axes.flat, loss_types)):
        ax.plot(
            loss_history[key],
            linewidth=2.5,
            color=colors[idx],
            marker="o",
            markersize=3,
            markevery=max(1, len(loss_history[key]) // 20),
        )

        ax.set_title(key, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss" if "loss" in key.lower() else "Value", fontsize=10)

        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cccccc")

        if len(loss_history[key]) > 0:
            final_val = loss_history[key][-1]
            ax.annotate(
                f"{final_val:.4f}",
                xy=(len(loss_history[key]) - 1, final_val),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                color="#666666",
            )

    for ax in axes.flat[n_plots:]:
        ax.set_visible(False)

    plt.suptitle("Training Metrics Dashboard", fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(
        Path(output_dir) / "loss_curves.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close()


def build_optimizer(params, learner, lr, weight_decay):
    if learner.lower() == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif learner.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif learner.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        return optimizer
    elif learner.lower() == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif learner.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        logger.warning("Received unrecognized optimizer, set default Adam optimizer")
        return torch.optim.Adam(params, lr=lr)


def get_scheduler(optimizer, lr_scheduler_type, warmup_steps, max_steps):
    if lr_scheduler_type.lower() == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        return get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=warmup_steps
        )


def save_checkpoint(
    args,
    model,
    optimizer,
    epoch,
    collision_rate,
    precision,
    recall,
    ndcg,
    output_dir,
    ckpt_file=None,
):
    # 保存模型参数到pth文件
    ckpt_path = os.path.join(output_dir, ckpt_file)
    state = {
        "args": args,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, ckpt_path, pickle_protocol=4)
    logger.info(f"Saving model checkpoint: {ckpt_path}")

    # 保存验证集结果到json文件
    json_path = os.path.join(output_dir, "best_validation_results.json")
    validation_results = {
        "epoch": epoch,
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg,
        "collision_rate": collision_rate,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=4, ensure_ascii=False)
    logger.info(f"Saving validation results: {json_path}")

    return ckpt_path


def check_nan(loss):
    if torch.isnan(loss):
        raise ValueError("Training loss is nan")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Joint RQVAE and LightGCN model")

    parser.add_argument(
        "--interaction_path",
        type=str,
        required=True,
        help="Path to user-item interaction data JSON file",
    )
    parser.add_argument(
        "--emb_path",
        type=str,
        required=True,
        help="Path to item embedding data NPY file",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.2, help="Test set ratio (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--sid_type",
        type=str,
        default="cf",
        help="SID type (default: cf)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size (default: 1024)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay (default: 0.001)",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=5,
        help="Early stop patience (default: 5)",
    )
    parser.add_argument(
        "--learner", type=str, default="adamw", help="Optimizer type (default: adam)"
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="constant",
        help="Learning rate scheduler type (default: constant)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=50,
        help="Number of warmup epochs (default: 0)",
    )
    parser.add_argument(
        "--eval_step", type=int, default=50, help="Evaluation step (default: 10)"
    )
    parser.add_argument(
        "--save_limit",
        type=int,
        default=5,
        help="Maximum number of saved checkpoints (default: 5)",
    )

    parser.add_argument(
        "--num_emb_list",
        type=int,
        nargs="+",
        default=[256, 256, 256],
        help="Number of embeddings per layer (default: [256, 256, 256])",
    )
    parser.add_argument(
        "--e_dim", type=int, default=32, help="Embedding dimension (default: 32)"
    )
    parser.add_argument(
        "--rqvae_layers",
        type=int,
        nargs="+",
        default=[2048, 1024, 512, 256, 128, 64],
        help="RQVAE hidden layers (default: [2048, 1024, 512, 256, 128, 64])",
    )
    parser.add_argument(
        "--dropout_prob",
        type=float,
        default=0.0,
        help="RQVAE dropout probability (default: 0.0)",
    )
    parser.add_argument(
        "--bn", action="store_true", help="Use batch normalization in RQVAE"
    )
    parser.add_argument(
        "--loss_type", type=str, default="mse", help="RQVAE loss type (default: mse)"
    )
    parser.add_argument(
        "--quant_loss_weight",
        type=float,
        default=1.0,
        help="RQVAE quantization loss weight (default: 1.0)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.25, help="RQVAE beta parameter (default: 0.25)"
    )
    parser.add_argument(
        "--kmeans_init", action="store_true", help="Use kmeans initialization in RQVAE"
    )
    parser.add_argument(
        "--kmeans_iters", type=int, default=100, help="Kmeans iterations (default: 100)"
    )
    parser.add_argument(
        "--sk_epsilons",
        type=float,
        nargs="+",
        default=[0.0, 0.0, 0.0],
        help="Sinkhorn epsilon values (default: [0.0, 0.0, 0.0])",
    )
    parser.add_argument(
        "--sk_iters", type=int, default=50, help="Sinkhorn iterations (default: 50)"
    )

    parser.add_argument(
        "--latent_dim_rec",
        type=int,
        default=32,
        help="LightGCN latent dimension (default: 64)",
    )
    parser.add_argument(
        "--lightGCN_n_layers",
        type=int,
        default=3,
        help="LightGCN number of layers (default: 3)",
    )
    parser.add_argument(
        "--keep_prob",
        type=float,
        default=0.6,
        help="LightGCN keep probability (default: 0.6)",
    )
    parser.add_argument(
        "--A_split", action="store_true", help="Use split adjacency matrix in LightGCN"
    )
    parser.add_argument(
        "--pretrain", type=int, default=0, help="LightGCN pretrain flag (default: 0)"
    )
    parser.add_argument(
        "--lgcn_dropout",
        type=float,
        default=0.0,
        help="LightGCN dropout (default: 0.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save model checkpoints and logs",
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    return args


def main():

    args = parse_args()

    if args.sid_type == "cf":
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]
        from model import LightGCN

        sys.argv = original_argv

    config = {
        "rqvae": {
            "num_emb_list": args.num_emb_list,
            "e_dim": args.e_dim,
            "layers": args.rqvae_layers,
            "dropout_prob": args.dropout_prob,
            "bn": args.bn,
            "loss_type": args.loss_type,
            "quant_loss_weight": args.quant_loss_weight,
            "beta": args.beta,
            "kmeans_init": args.kmeans_init,
            "kmeans_iters": args.kmeans_iters,
            "sk_epsilons": args.sk_epsilons,
            "sk_iters": args.sk_iters,
        },
        "lightgcn": {
            "latent_dim_rec": args.latent_dim_rec,
            "lightGCN_n_layers": args.lightGCN_n_layers,
            "keep_prob": args.keep_prob,
            "A_split": args.A_split,
            "pretrain": args.pretrain,
            "dropout": args.lgcn_dropout,
        },
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "early_stop_patience": args.early_stop_patience,
        },
        "test": {
            "batch_size": args.batch_size,
        },
    }

    dataset = JointEmbInterDataset(
        interaction_path=args.interaction_path,
        emb_path=args.emb_path,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split=args.A_split,
        folds=100,
    )

    model = JointCFRQVAE(config, dataset, sid_type=args.sid_type)
    model = model.cuda()

    optimizer = build_optimizer(
        model.parameters(),
        args.learner,
        config["train"]["lr"],
        config["train"]["weight_decay"],
    )

    data_num = (dataset.embeddings.shape[0] - 1) // config["train"]["batch_size"] + 1
    warmup_steps = args.warmup_epochs * data_num
    max_steps = args.epochs * data_num

    scheduler = get_scheduler(
        optimizer,
        args.lr_scheduler_type,
        warmup_steps,
        max_steps,
    )
    loss_history = defaultdict(list)

    best_stop_score = 0
    best_collision_rate = np.inf
    patience = config["train"]["early_stop_patience"]
    counter = 0

    pbar = tqdm(range(config["train"]["epochs"]), ncols=100, desc="Train")

    for epoch in pbar:
        model.train()

        total_loss = 0
        total_recon_loss = 0
        total_quant_loss = 0

        for _ in range(data_num):
            optimizer.zero_grad()

            users, pos_items = dataset.get_train_batch(config["train"]["batch_size"])
            neg_items = dataset.get_neg_items(users.numpy(), pos_items.numpy())

            users = users.cuda()
            pos_items = pos_items.cuda()
            neg_items = neg_items.cuda()

            batch_loss, loss_dict = model(users, pos_items, neg_items)

            check_nan(batch_loss)

            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.rqvae.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += batch_loss.item()
            total_recon_loss += loss_dict["rq_loss_recon_loss"]
            total_quant_loss += loss_dict["rq_quant_loss"]

        avg_loss = total_loss / data_num
        avg_recon_loss = total_recon_loss / data_num
        avg_quant_loss = total_quant_loss / data_num

        loss_history["total_loss"].append(avg_loss)
        loss_history["rq_loss_recon_loss"].append(avg_recon_loss)
        loss_history["rq_quant_loss"].append(avg_quant_loss)
        loss_history["cf_bpr_loss"].append(loss_dict["cf_bpr_loss"])
        loss_history["cf_reg_loss"].append(loss_dict["cf_reg_loss"])
        loss_history["align_loss"].append(loss_dict["align_loss"])

        pbar.set_description(
            f"Epoch [{epoch+1}/{config['train']['epochs']}] Loss: {avg_loss:.4f}"
        )

        if (epoch + 1) % args.eval_step == 0:
            precision, recall, ndcg, collision_rate = Test(dataset, model, config)
            # logger.info(f"Test - Precision: {precision:.4f}, Recall: {recall:.4f}, NDCG: {ndcg:.4f}, Collision Rate: {collision_rate:.4f}")
            loss_history["val_precision"].append(precision)
            loss_history["val_recall"].append(recall)
            loss_history["val_ndcg"].append(ndcg)
            loss_history["val_collision_rate"].append(collision_rate)

            if args.sid_type == "cf":
                stop_score = ndcg / max(collision_rate, 1e-5)
            else:
                stop_score = 1 / max(collision_rate, 1e-5)

            plot_loss_curves(loss_history, args.sid_type, args.output_dir)

            if stop_score > best_stop_score:
                best_stop_score = stop_score
                counter = 0
                save_checkpoint(
                    args,
                    model,
                    optimizer,
                    epoch,
                    collision_rate,
                    precision,
                    recall,
                    ndcg,
                    args.output_dir,
                    "best_model.pth",
                )
            else:
                counter += 1
                # logger.info(f"Early stop counter: {counter}/{patience}")

            if counter >= patience:
                logger.info("Early stopping triggered")
                break

    final_precision, final_recall, final_ndcg, final_collision_rate = Test(
        dataset, model, config
    )
    logger.info(
        f"Final Test - Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, NDCG: {final_ndcg:.4f}, Collision Rate: {final_collision_rate:.4f}"
    )

    plot_loss_curves(loss_history, args.sid_type, args.output_dir)


if __name__ == "__main__":
    main()
