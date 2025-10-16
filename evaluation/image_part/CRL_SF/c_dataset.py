import os
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class PairedUnsupervisedImageDataset(Dataset):

    def __init__(self, meta_csv_path: str, transform=None):
        self.df = pd.read_csv(meta_csv_path)
        self.root = os.path.dirname(meta_csv_path)

        self.grouped = self.df.groupby('id').groups
        self.ids = list(self.grouped.keys())

        self.z_fields = [c for c in self.df.columns
                         if c not in ('id', 'view', 'render_path')]

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        indices   = list(self.grouped[sample_id])
        assert len(indices) == 4, f"id={sample_id} 下不止 4 个 view"


        i1, i2, i3, i4 = random.sample(indices, 4)
        rows = [self.df.iloc[i] for i in (i1, i2, i3, i4)]


        imgs = []
        views = []
        zs   = []
        for row in rows:
            path = os.path.join(self.root, row['render_path'])
            img  = Image.open(path).convert('RGB')
            imgs.append(self.transform(img))
            views.append(int(row['view']))
            z_arr = row[self.z_fields].to_numpy(dtype=np.float32)
            zs.append(torch.from_numpy(z_arr))


        pair_imgs  = ((imgs[0], imgs[1]), (imgs[2], imgs[3]))
        pair_views = ((views[0], views[1]), (views[2], views[3]))
        pair_zs    = ((zs[0],    zs[1]),    (zs[2],    zs[3]))

        return sample_id, pair_imgs, pair_views, pair_zs
