"""
FER2013 dataset loader (msambare/fer2013 on Kaggle):
Assumes structure like:
root/
  train/
    Angry/
    Disgust/
    Fear/
    Happy/
    Neutral/
    Sad/
    Surprise/
  test/
    ...same classes...
"""
import os, random
from typing import Tuple, Optional, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]

def _collect_images(root_dir: str) -> List[Tuple[str,int]]:
    pairs = []
    for idx, cls in enumerate(EMOTIONS):
        d = os.path.join(root_dir, cls)
        if not os.path.isdir(d): 
            continue
        for fn in os.listdir(d):
            if fn.lower().endswith((".jpg",".jpeg",".png")):
                pairs.append((os.path.join(d, fn), idx))
    return pairs

class FERFolder(Dataset):
    def __init__(self, root_split: str, split: str = "train", input_size: int = 224, augment: bool = True,
                 file_list: Optional[List[Tuple[str,int]]] = None):
        """
        root_split: path to train/ or test/ folder
        """
        self.input_size = input_size
        if file_list is None:
            self.items = _collect_images(root_split)
        else:
            self.items = file_list

        if augment and split == "train":
            self.tf = T.Compose([
                T.Resize((input_size, input_size)),
                T.RandomHorizontalFlip(0.5),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        img = Image.open(path)
        # images are in grayscale, need to convert to RGB for RESNET/MOBILENET
        img = img.convert("RGB")
        x = self.tf(img)
        return x, torch.tensor(int(y), dtype=torch.long)
