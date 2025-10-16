import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
import torch

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


