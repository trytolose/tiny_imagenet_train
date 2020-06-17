from torch.utils.data.dataset import Dataset
import torch
import os
from typing import Callable, List, Optional, Sequence, Union
from PIL import Image
from dataclasses import dataclass, replace
from pathlib import Path
from pandas import DataFrame
import pandas as pd
from torch.utils.data.dataloader import default_collate
import cv2
from torchvision import transforms
from utils.autoaugment import ImageNetPolicy

Transform = Callable[[Image.Image], Image.Image]
DATA_ROOT = Path("../input/tiny-imagenet-200")
TRAIN_PATH = DATA_ROOT / "train"
VAL_PATH = DATA_ROOT / "val"
ALL_FOLDERS = [
    dir_name
    for r, d, f in os.walk(TRAIN_PATH)
    for dir_name in d
    if dir_name != "images"
]
FOLDERS_TO_NUM = {val: index for index, val in enumerate(ALL_FOLDERS)}

LABELS = pd.read_csv(
    DATA_ROOT / "words.txt", sep="\t", header=None, index_col=0)[1].to_dict()
VAL_LABELS = pd.read_csv(
    DATA_ROOT / "val" / "val_annotations.txt", sep="\t", header=None, index_col=0)[1].to_dict()


class TinyImagenetDataset(Dataset):

    _root: Path
    _df: DataFrame
    _auto_transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(), ImageNetPolicy(), 
                transforms.ToTensor(), transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])])

    def __init__(self, path, transform):
        self._transform = transform
        if not os.path.isdir(path):
            raise NotADirectoryError(f"{path} is not a directory.")
        all_files = [
            os.path.join(r, fyle)
            for r, d, f in os.walk(path)
            for fyle in f
            if ".JPEG" in fyle
        ]
        labels = [
            FOLDERS_TO_NUM.get(
                os.path.basename(f).split("_")[0],
                FOLDERS_TO_NUM.get(VAL_LABELS.get(os.path.basename(f))),
            )
            for f in all_files
        ]
        self._df = pd.DataFrame({"path": all_files, "label": labels})
        

    def __getitem__(self, index: int):
        path, label = self._df.loc[index, :]
        
        if self._transform:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            data = {"image": image}
            augmented = self._transform(**data)
            return augmented['image'], label
        else:

            image = Image.open(path).convert("RGB")
            image = self._auto_transform(image)
            return image, label


    def __len__(self) -> int:
        return len(self._df)


