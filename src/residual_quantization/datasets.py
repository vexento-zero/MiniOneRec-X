import numpy as np
import torch
import json
import torch.utils.data as data
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


class EmbDataset(data.Dataset):

    def __init__(self, data_path):

        self.data_path = data_path
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)
        self.embeddings = np.load(data_path)

        # Check for NaN values and handle them
        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in embeddings")
            # Replace NaN with zeros
            self.embeddings[nan_mask] = 0.0

        # Check for infinite values
        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in embeddings")
            # Replace inf with zeros
            self.embeddings[inf_mask] = 0.0

        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(
            f"Embeddings stats - min: {self.embeddings.min():.6f}, max: {self.embeddings.max():.6f}, mean: {self.embeddings.mean():.6f}"
        )

        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)


class JointEmbInterDataset:
    def __init__(
        self,
        interaction_path,
        emb_path,
        test_ratio=0.2,
        seed=42,
        split=False,
        folds=100,
    ):
        self.Graph = None
        self.path = interaction_path
        self.split = split
        self.folds = folds

        self.embeddings = np.load(emb_path)
        nan_mask = np.isnan(self.embeddings)
        if nan_mask.any():
            print(f"Warning: Found {nan_mask.sum()} NaN values in embeddings")
            self.embeddings[nan_mask] = 0.0

        inf_mask = np.isinf(self.embeddings)
        if inf_mask.any():
            print(f"Warning: Found {inf_mask.sum()} infinite values in embeddings")
            self.embeddings[inf_mask] = 0.0

        print(f"Loaded embeddings shape: {self.embeddings.shape}")
        print(
            f"Embeddings stats - min: {self.embeddings.min():.6f}, max: {self.embeddings.max():.6f}, mean: {self.embeddings.mean():.6f}"
        )
        self.embeddings = np.nan_to_num(
            self.embeddings, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.emb_dim = self.embeddings.shape[-1]
        self.num_embeddings = len(self.embeddings)

        self.m_items = self.num_embeddings

        with open(interaction_path, "r") as f:
            raw_data = json.load(f)

        user_items = {}
        self.n_user = 0
        self.max_item_id = 0
        for uid, items in raw_data.items():
            uid = int(uid)
            items = [int(i) for i in items]
            user_items[uid] = items
            self.n_user = max(self.n_user, uid)
            self.max_item_id = max(self.max_item_id, max(items))
        self.n_user += 1

        actual_num_items = self.max_item_id + 1
        print(
            f"Debug: Max item ID = {self.max_item_id}, Number of embeddings = {self.num_embeddings}"
        )
        print(f"Debug: Actual number of items in interaction data = {actual_num_items}")

        self.m_items = actual_num_items

        if self.m_items > self.num_embeddings:
            raise ValueError(
                f"Number of items in interaction data ({self.m_items}) exceeds number of embeddings ({self.num_embeddings}). Please ensure the embedding file contains embeddings for all items."
            )

        random.seed(seed)
        self.train_interactions = []
        self.test_interactions = []

        for uid, items in user_items.items():
            if len(items) < 2:
                for iid in items:
                    self.train_interactions.append((uid, iid))
            else:
                test_idx = random.randint(0, len(items) - 1)
                for i, iid in enumerate(items):
                    if i == test_idx:
                        self.test_interactions.append((uid, iid))
                    else:
                        self.train_interactions.append((uid, iid))

        self.trainUser = np.array([x[0] for x in self.train_interactions])
        self.trainItem = np.array([x[1] for x in self.train_interactions])
        self.testUser = np.array([x[0] for x in self.test_interactions])
        self.testItem = np.array([x[1] for x in self.test_interactions])

        if np.any(self.trainItem >= self.m_items):
            invalid_ids = np.unique(self.trainItem[self.trainItem >= self.m_items])
            raise ValueError(
                f"Found invalid item IDs in training data: {invalid_ids}. Maximum valid item ID is {self.m_items - 1}."
            )

        if np.any(self.testItem >= self.m_items):
            invalid_ids = np.unique(self.testItem[self.testItem >= self.m_items])
            raise ValueError(
                f"Found invalid item IDs in test data: {invalid_ids}. Maximum valid item ID is {self.m_items - 1}."
            )

        self.traindataSize = len(self.train_interactions)
        self.testDataSize = len(self.test_interactions)

        self.UserItemNet = sp.csr_matrix(
            (np.ones(self.traindataSize), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_items),
        )

        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()

        print(f"Users: {self.n_user}, Items: {self.m_items}")
        print(f"Max item ID in interaction data: {self.max_item_id}")
        print(f"Train: {self.traindataSize}, Test: {self.testDataSize}")

    @property
    def n_users(self):
        return self.n_user

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def __build_test(self):
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def get_neg_items(self, users, pos_items, num_neg=1):
        neg_items = []
        for i, user in enumerate(users):
            pos_set = set(self._allPos[user])
            neg = np.random.randint(0, self.m_items)
            while neg in pos_set:
                neg = np.random.randint(0, self.m_items)
            assert 0 <= neg < self.m_items, f"Negative item {neg} is out of bounds!"
            neg_items.append(neg)
        return torch.LongTensor(neg_items)

    def get_train_batch(self, batch_size):
        idx = np.random.randint(0, self.traindataSize, batch_size)
        users = self.trainUser[idx]
        items = self.trainItem[idx]
        return torch.LongTensor(users), torch.LongTensor(items)

    def __len__(self):
        return self.traindataSize

    def __getitem__(self, idx):
        return self.trainUser[idx], self.trainItem[idx]

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            end = (
                (i_fold + 1) * fold_len
                if i_fold != self.folds - 1
                else self.n_users + self.m_items
            )
            A_fold.append(
                self._convert_sp_mat_to_sp_tensor(A[start:end])
                .coalesce()
                .to(torch.device("cuda"))
            )
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(Path(self.path).parent / "s_pre_adj_mat.npz")
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time.time()
                adj_mat = sp.dok_matrix(
                    (self.n_users + self.m_items, self.n_users + self.m_items),
                    dtype=np.float32,
                )
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[: self.n_users, self.n_users :] = R
                adj_mat[self.n_users :, : self.n_users] = R.T
                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(Path(self.path).parent / "s_pre_adj_mat.npz", norm_adj)

            if self.split:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(torch.device("cuda"))
                print("don't split the matrix")
        return self.Graph
