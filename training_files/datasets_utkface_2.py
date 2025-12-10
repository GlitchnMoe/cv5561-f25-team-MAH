import os, re, random
from typing import List, Tuple, Optional, Sequence, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

UTK_PATTERN = re.compile(r"^(\d+)_(\d+)_\d+_.*\.(jpg|jpeg|png)$", re.IGNORECASE)

def parse_utk_filename(fname: str):
    m = UTK_PATTERN.match(fname)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))  # age, gender

def _scan_roots(roots: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(roots, (list, tuple, set)):
        dirs = list(roots)
    else:
        dirs = [roots]
    files = []
    for r in dirs:
        for f in os.listdir(r):
            if UTK_PATTERN.match(f):
                files.append(os.path.join(r, f))  # store full path
    return files

class UTKFaceDataset(Dataset):
    def __init__(
        self,
        root: Union[str, Sequence[str]],
        split: str = "train",
        split_ratio=(0.9, 0.1),
        seed: int = 42,
        input_size: int = 224,
        task: str = "both",
        augment: bool = True,
        file_list: Optional[List[str]] = None,
    ):
        """
        root: str or list of dirs (e.g. ["./part1","./part2","./part3"])
        split: "train", "val", or "test"
        split_ratio:
          - length 2: (train, val)
          - length 3: (train, val, test)
        task: "age", "gender", or "both"
        """
        assert task in {"age", "gender", "both"}
        self.task = task
        self.input_size = input_size

        if file_list is None:
            all_files = _scan_roots(root)
            random.Random(seed).shuffle(all_files)
            n = len(all_files)

            if len(split_ratio) == 2:
                r_train, r_val = split_ratio
                n_train = int(r_train * n)
                train_files = all_files[:n_train]
                val_files = all_files[n_train:]

                if split == "train":
                    self.files = train_files
                else:
                    # anything not "train" (val, test, etc.) -> val_files
                    self.files = val_files

            elif len(split_ratio) == 3:
                r_train, r_val, r_test = split_ratio
                n_train = int(r_train * n)
                n_val = int(r_val * n)
                train_files = all_files[:n_train]
                val_files = all_files[n_train:n_train + n_val]
                test_files = all_files[n_train + n_val:]

                if split == "train":
                    self.files = train_files
                elif split == "val":
                    self.files = val_files
                elif split == "test":
                    self.files = test_files
                else:
                    raise ValueError(f"Unknown split '{split}', expected 'train', 'val', or 'test'")
            else:
                raise ValueError("split_ratio must have length 2 or 3")
        else:
            # file_list should contain full paths
            self.files = file_list

        if augment and split == "train":
            self.tf = T.Compose([
                T.Resize((input_size, input_size)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]  # full path
        fname = os.path.basename(path)
        labels = parse_utk_filename(fname)
        while labels is None:
            idx = (idx + 1) % len(self.files)
            path = self.files[idx]
            fname = os.path.basename(path)
            labels = parse_utk_filename(fname)
        age, gender = labels
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        if self.task == "age":
            return x, torch.tensor(float(age), dtype=torch.float32)
        elif self.task == "gender":
            return x, torch.tensor(int(gender), dtype=torch.long)
        else:
            return (
                x,
                torch.tensor(float(age), dtype=torch.float32),
                torch.tensor(int(gender), dtype=torch.long),
            )
