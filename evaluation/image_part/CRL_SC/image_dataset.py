import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
import torch
import random
import numpy as np
from torch.utils.data import Sampler, DataLoader
class UnsupervisedImageDataset(Dataset):
    """
    A PyTorch Dataset for unsupervised learning on images with associated metadata (Z).
    
    Returns a 4-tuple per sample: (id, image, view, z).
    """
    def __init__(
        self,
        meta_csv_path: str,
        transform=None
    ):
        """
        Args:
            meta_csv_path (str): Path to the metadata CSV file.
            transform (callable, optional): A function/transform that takes in a PIL image
                and returns a torch tensor of shape [C, H, W]. If None, a default
                Resize+ToTensor transform to (224,224) is applied.
        """
        self.meta_csv_path = meta_csv_path
        self.df = pd.read_csv(meta_csv_path)
        self.root = os.path.dirname(meta_csv_path)
        # Identify Z fields: all columns except id, view, render_path
        self.z_fields = [col for col in self.df.columns if col not in ('id', 'view', 'render_path')]
        
        from torchvision import transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = int(row['id'])
        view = int(row['view'])
        img_path = os.path.join(self.root, row['render_path'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Z: array of metadata fields (h, r, m, k, l, etc.)
        z_array = row[self.z_fields].to_numpy(dtype=np.float32)
        z = torch.from_numpy(z_array)
        
        return sample_id, image, view, z




    def __init__(self, dataset, batch_size):
        # assert batch_size % 4 == 0
        self.batch_size = batch_size
        self.per_view  = batch_size // 4

        if isinstance(dataset, Subset):
            self.base_dataset = dataset.dataset
            abs_indices = list(dataset.indices)
        else:
            self.base_dataset = dataset
            abs_indices = list(range(len(dataset)))


        self.abs2rel = {abs_idx: rel_idx
                        for rel_idx, abs_idx in enumerate(abs_indices)}


        df = self.base_dataset.df
        views = sorted(df['view'].unique())



        self.view_to_rel = {v: [] for v in views}
        for abs_idx in abs_indices:
            v = int(df.iloc[abs_idx]['view'])
            rel_idx = self.abs2rel[abs_idx]
            self.view_to_rel[v].append(rel_idx)


        self.num_batches = min(
            len(idxs) // self.per_view
            for idxs in self.view_to_rel.values()
        )

    def __iter__(self):

        for idxs in self.view_to_rel.values():
            random.shuffle(idxs)


        for batch_idx in range(self.num_batches):
            batch = []
            for rel_idxs in self.view_to_rel.values():
                start = batch_idx * self.per_view
                end   = start + self.per_view
                batch.extend(rel_idxs[start:end])
            yield batch  

    def __len__(self):
        return self.num_batches



def split_dataset(dataset: UnsupervisedImageDataset, key_fn):
    """
    Splits a dataset into subsets based on a key function applied to each DataFrame row.

    Args:
        dataset (UnsupervisedImageDataset): The dataset to split.
        key_fn (callable): Function taking a pd.Series (row) and returning a grouping key.

    Returns:
        Dict[key, torch.utils.data.Subset]: Mapping each key to a Subset of the dataset.
    """
    groups = {}
    for idx, row in dataset.df.iterrows():
        key = key_fn(row)
        groups.setdefault(key, []).append(idx)
    subsets = {key: Subset(dataset, indices) for key, indices in groups.items()}
    return subsets

# Example: split by the 'view' field into 4 subsets
def split_by_view(dataset: UnsupervisedImageDataset):
    return split_dataset(dataset, key_fn=lambda row: int(row['view']))

# Example usage:
# ds = UnsupervisedImageDataset("/mnt/disk2/.../Spring.csv")
# splits = split_by_view(ds)
# splits[0], splits[1], splits[2], splits[3]

