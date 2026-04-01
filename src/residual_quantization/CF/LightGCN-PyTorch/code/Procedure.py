"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
"""

import world
import numpy as np
import torch
import utils
from utils import timer
import model
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
from rich_logger import logger
import json

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, pbar):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config["bpr_batch_size"] + 1
    aver_loss = 0.0
    losses = []
    for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(
        utils.minibatch(
            users, posItems, negItems, batch_size=world.config["bpr_batch_size"]
        )
    ):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        losses.append(cri)
        pbar.update(1)
    aver_loss = aver_loss / total_batch
    timer.zero()
    return losses


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    result_dict = {}
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        # pre.append(ret["precision"])
        # recall.append(ret["recall"])
        # ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        result_dict[f"Top@{k}"] = {
            "recall": ret["recall"],
            "precision": ret["precision"],
            "ndcg": utils.NDCGatK_r(groundTrue, r, k),
        }

    return result_dict


def Test(dataset, Recmodel):
    u_batch_size = world.config["test_u_batch_size"]
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(
                f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}"
            )
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in tqdm(
            utils.minibatch(users, batch_size=u_batch_size),
            ncols=100,
            desc=f"test",
            total=total_batch,
        ):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)

            # rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [
            #         utils.AUC(rating[i],
            #                   dataset,
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        result_dict = defaultdict(lambda: defaultdict(list))
        for x in X:
            _result_dict = test_one_batch(x)
            for topk in _result_dict:
                for metric in _result_dict[topk]:
                    result_dict[topk][metric].append(_result_dict[topk][metric])
        for topk in result_dict:
            for metric in result_dict[topk]:
                result_dict[topk][metric] = np.mean(result_dict[topk][metric])

        # print()
        # logger.info(f"result_dict = {json.dumps(result_dict, indent=4)}")
        return result_dict
