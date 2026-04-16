import csv
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data


STANDARD_SEQUENCES = [
    "T1",
    "T2",
    "ADC",
    "DWI",
    "DCE1",
    "DCE2",
    "DCE3",
    "DCE4",
    "DCE5",
    "DCE6",
    "DCE7",
]


class ANCAdataset(data.Dataset):
    def __init__(
        self,
        root,
        csv_path,
        TrainValTest="train",
        split_seed=3407,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        sequences=None,
        max_slices=96,
    ):
        self.root = root
        self.csv_path = csv_path
        self.TrainValTest = TrainValTest
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.sequences = sequences or STANDARD_SEQUENCES
        self.max_slices = max_slices
        self.name = None
        self.file_id = None

        self.dataset_mode = self._detect_dataset_mode(csv_path)
        self.has_tabular_features = self.dataset_mode != "standardized_mri"
        if self.dataset_mode == "standardized_mri":
            self.items = self._build_standardized_items()
        else:
            self.items = self._build_legacy_items()

    def __getitem__(self, index):
        item = self.items[index]
        self.file_id = item["patient_id"]
        self.name = item["name"]

        if self.dataset_mode == "standardized_mri":
            feature_data = torch.load(item["feature_path"], map_location="cpu")
            img_data = {}
            inner_slice_mask = {}
            inter_slice_mask = {}

            for seq in self.sequences:
                if seq not in feature_data:
                    continue

                seq_data = feature_data[seq]
                pe = seq_data["patch_embed"].float()
                ma = seq_data["mask_attention"].float()
                sw = seq_data["slices_weight"].float()

                # Uniformly subsample if exceeding max_slices
                n = pe.shape[0]
                if self.max_slices and n > self.max_slices:
                    indices = torch.linspace(0, n - 1, self.max_slices).long()
                    pe = pe[indices]
                    ma = ma[indices]
                    sw = sw[indices]

                img_data[seq] = pe
                inner_slice_mask[seq] = ma
                inter_slice_mask[seq] = sw

            x_categ = torch.zeros(4, dtype=torch.long)
            x_numer = torch.zeros(23, dtype=torch.float32)

            return (
                torch.tensor(item["label"], dtype=torch.long),
                x_categ,
                x_numer,
                img_data,
                inner_slice_mask,
                inter_slice_mask,
            )

        tensor_data = torch.load(item["feature_path"], map_location="cpu")

        img_data = tensor_data["patch_embed"].float()
        inner_slice_mask = tensor_data["mask_attention"].float()
        inter_slice_mask = tensor_data["slices_weight"].float()

        tabel_data = item["tabular_data"]
        x_categ = torch.tensor(tabel_data[-9:]).float()
        x_categ = torch.nonzero(x_categ == 1).flatten() - torch.tensor([0, 3, 5, 7])
        x_numer = torch.tensor(tabel_data[:-9]).float()

        return (
            torch.tensor(item["label"], dtype=torch.long),
            x_categ,
            x_numer,
            img_data,
            inner_slice_mask,
            inter_slice_mask,
        )

    def __len__(self):
        return len(self.items)

    def _detect_dataset_mode(self, csv_path):
        with open(csv_path, "r", newline="") as file:
            reader = csv.reader(file)
            first_row = next(reader)

        if len(first_row) >= 2 and first_row[0] == "patient_id" and first_row[1] == "label":
            return "standardized_mri"
        return "legacy"

    def _build_standardized_items(self):
        with open(self.csv_path, "r", newline="") as file:
            reader = csv.DictReader(file)
            rows = [row for row in reader]

        valid_rows = []
        for row in rows:
            feature_path = os.path.join(self.root, f"{row['patient_id']}.pth")
            if not os.path.exists(feature_path):
                continue

            valid_rows.append(
                {
                    "patient_id": row["patient_id"],
                    "name": row["patient_id"],
                    "label": int(row["label"]),
                    "feature_path": feature_path,
                }
            )

        split_items = self._stratified_split(valid_rows)
        return split_items[self.TrainValTest]

    def _stratified_split(self, items):
        label_groups = defaultdict(list)
        for item in items:
            label_groups[item["label"]].append(item)

        rng = random.Random(self.split_seed)
        split_map = {"train": [], "val": [], "test": []}

        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if total_ratio <= 0:
            raise ValueError("train_ratio + val_ratio + test_ratio must be positive")

        train_ratio = self.train_ratio / total_ratio
        val_ratio = self.val_ratio / total_ratio

        for label_items in label_groups.values():
            label_items = sorted(label_items, key=lambda item: item["patient_id"])
            rng.shuffle(label_items)

            num_items = len(label_items)
            num_train = int(round(num_items * train_ratio))
            num_val = int(round(num_items * val_ratio))

            if num_train + num_val > num_items:
                overflow = num_train + num_val - num_items
                num_val = max(0, num_val - overflow)

            num_test = num_items - num_train - num_val

            split_map["train"].extend(label_items[:num_train])
            split_map["val"].extend(label_items[num_train:num_train + num_val])
            split_map["test"].extend(label_items[num_train + num_val:num_train + num_val + num_test])

        for split_name in split_map:
            split_map[split_name] = sorted(split_map[split_name], key=lambda item: item["patient_id"])

        return split_map

    def _build_legacy_items(self):
        if self.TrainValTest == "train":
            train_mark = 1
        elif self.TrainValTest == "val":
            train_mark = 0
        elif self.TrainValTest == "test":
            train_mark = 2
        else:
            raise ValueError(f"Unsupported split: {self.TrainValTest}")

        items = []
        with open(self.csv_path, "r", newline="") as file:
            file_reader = csv.reader(file)
            for line in file_reader:
                if int(line[3]) != train_mark:
                    continue

                file_path = os.path.join(self.root, line[0])
                if not os.path.isdir(file_path):
                    continue

                file_list = sorted(os.listdir(file_path))
                if not file_list:
                    continue

                tabular_data = np.stack([float(value) for value in line[4:]])
                items.append(
                    {
                        "patient_id": line[0],
                        "name": line[1],
                        "label": int(line[2]),
                        "tabular_data": tabular_data,
                        "feature_path": os.path.join(file_path, file_list[0]),
                    }
                )

        return items

    def getFileName(self):
        return self.name

    def getFileId(self):
        return self.file_id
