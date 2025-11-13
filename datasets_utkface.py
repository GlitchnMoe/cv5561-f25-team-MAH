import os, re, random
from typing import List, Tuple, Optional, Sequence, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

UTK_PATTERN = re.compile(r"^(\d+)_(\d+)_\d+_.*\.(jpg|jpeg|png)$", re.IGNORECASE)

def parse_utk_filename(fname: str):
    m = UTK_PATTERN.match(fname)
    if not m: return None
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
    def __init__(self, root: Union[str, Sequence[str]], split: str = "train", split_ratio=(0.9, 0.1), seed: int = 42,
                 input_size: int = 224, task: str = "both", augment: bool = True, file_list: Optional[List[str]] = None):
        assert task in {"age", "gender", "both"}
        self.task = task
        self.input_size = input_size

        if file_list is None:
            all_files = _scan_roots(root)
            random.Random(seed).shuffle(all_files)
            n = len(all_files); n_train = int(split_ratio[0] * n)
            train_files = all_files[:n_train]; val_files = all_files[n_train:]
            self.files = train_files if split == "train" else val_files
        else:
            # file_list should contain full paths
            self.files = file_list

        if augment and split == "train":
            self.tf = T.Compose([
                T.Resize((input_size, input_size)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]                         # full path
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
            return x, torch.tensor(float(age), dtype=torch.float32), torch.tensor(int(gender), dtype=torch.long)
