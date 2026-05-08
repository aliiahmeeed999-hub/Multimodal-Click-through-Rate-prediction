
# =========================================================================
# Copyright (C) 2024. The FuxiCTR Library. All rights reserved.
#
# Patched: float32/int32 stacking, gc, GPU pin_memory / workers, collate .long() for indices.
# =========================================================================

import gc
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import torch


class ParquetDataset(Dataset):
    def __init__(self, data_path):
        self.column_index = dict()
        self.darray = self.load_data(data_path)

    def __getitem__(self, index):
        return self.darray[index, :]

    def __len__(self):
        return self.darray.shape[0]

    def load_data(self, data_path):
        df = pd.read_parquet(data_path)
        data_arrays = []
        idx = 0
        try:
            for col in df.columns:
                if df[col].dtype == "object":
                    array = np.asarray(df[col].to_list(), dtype=np.int32)
                    if array.ndim == 1:
                        array = array.reshape(-1, 1)
                    seq_len = array.shape[1]
                    self.column_index[col] = [i + idx for i in range(seq_len)]
                    idx += seq_len
                else:
                    array = df[col].to_numpy()
                    if np.issubdtype(array.dtype, np.floating):
                        array = array.astype(np.float32, copy=False)
                    elif np.issubdtype(array.dtype, np.integer):
                        array = array.astype(np.int32, copy=False)
                    self.column_index[col] = idx
                    idx += 1
                data_arrays.append(array)
        finally:
            del df
            gc.collect()
        stacked = np.column_stack(data_arrays)
        return np.ascontiguousarray(stacked)


class MMCTRDataLoader(DataLoader):
    def __init__(self, feature_map, data_path, item_info, batch_size=32, shuffle=False,
                 num_workers=1, max_len=100, **kwargs):
        if not data_path.endswith(".parquet"):
            data_path += ".parquet"
        self.dataset = ParquetDataset(data_path)
        column_index = self.dataset.column_index
        gpu = int(kwargs.get("gpu", -1))
        use_cuda = gpu >= 0 and torch.cuda.is_available()
        pin_memory = bool(use_cuda)
        if use_cuda and num_workers == 0:
            num_workers = 2
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=BatchCollator(feature_map, max_len, column_index, item_info),
        )
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self):
        return self.num_batches


class BatchCollator(object):
    def __init__(self, feature_map, max_len, column_index, item_info):
        self.feature_map = feature_map
        self.item_info = pd.read_parquet(item_info)
        self.max_len = max_len
        self.column_index = column_index

    def __call__(self, batch):
        batch_tensor = default_collate(batch)
        all_cols = set(list(self.feature_map.features.keys()) + self.feature_map.labels)
        batch_dict = dict()
        for col, idx in self.column_index.items():
            if col in all_cols:
                batch_dict[col] = batch_tensor[:, idx]
        batch_seqs = batch_dict["item_seq"][:, -self.max_len:].long()
        del batch_dict["item_seq"]
        mask = (batch_seqs > 0).float()
        item_index = batch_dict["item_id"].long().numpy().reshape(-1, 1)
        del batch_dict["item_id"]
        batch_items = np.hstack([batch_seqs.numpy(), item_index]).astype(
            np.int64, copy=False
        ).flatten()
        item_info = self.item_info.iloc[batch_items]
        item_dict = dict()
        for col in item_info.columns:
            if col in all_cols:
                item_dict[col] = torch.from_numpy(np.array(item_info[col].to_list()))
        return batch_dict, item_dict, mask
