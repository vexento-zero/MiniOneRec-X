import world
import utils
import torch
import Procedure
from tqdm import tqdm
from rich_logger import logger
import sys
import numpy as np

sys.path.append("/ssd/hust-tangxi/workspace/MOR")
from src.utils.plot import plot_scores
from pathlib import Path

# ==============================
utils.set_seed(world.seed)
logger.info(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
logger.info(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(
            torch.load(weight_file, map_location=torch.device("cpu"))
        )
        logger.info(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        logger.info(f"{weight_file} not exists, start from beginning")
Neg_k = 1


pbar = tqdm(
    ncols=100,
    desc=f"Training LightGCN",
    total=world.TRAIN_epochs
    * (dataset.trainDataSize // world.config["bpr_batch_size"] + 1),
)
losses = []
ndcgs = []
best_ndcg = 0
best_epoch = 0
patience = 5  # 早停轮数
counter = 0

for epoch in range(world.TRAIN_epochs):
    if epoch % 10 == 0:
        val_res = Procedure.Test(dataset, Recmodel)
        current_ndcg = val_res["Top@20"]["ndcg"]
        ndcgs.append([current_ndcg])

        # 保存最佳模型
        if current_ndcg > best_ndcg:
            best_ndcg = current_ndcg
            best_epoch = epoch
            counter = 0
            torch.save(Recmodel.state_dict(), weight_file)
            print(f"Epoch {epoch}: New best NDCG = {current_ndcg:.4f}, model saved")
        else:
            counter += 1
            print(
                f"Epoch {epoch}: NDCG = {current_ndcg:.4f}, best = {best_ndcg:.4f}, patience {counter}/{patience}"
            )

        # 早停判断
        if counter >= patience:
            print(
                f"Early stopping at epoch {epoch}, best NDCG = {best_ndcg:.4f} at epoch {best_epoch}"
            )
            break

    _losses = Procedure.BPR_train_original(dataset, Recmodel, bpr, pbar)
    losses.extend([[x] for x in _losses])
    pbar.set_description(
        f"Training LightGCN [{epoch+1}/{world.TRAIN_epochs}]: {np.mean(_losses):.4f}"
    )
    plot_scores(losses, window=10, name=Path(world.PATH, f"loss_curve.png"))
    plot_scores(ndcgs, window=1, name=Path(world.PATH, f"ndcgs_curve.png"))

# 训练结束后加载最佳模型
if best_epoch > 0:
    print(f"Loading best model from epoch {best_epoch}")
    Recmodel.load_state_dict(torch.load(weight_file))
