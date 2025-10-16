import os
import re
from typing import List, Dict, Tuple, Optional, Any

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


def _safe_pil_to_tensor(img: Image.Image) -> Tensor:
    """
    Convert PIL.Image to a float32 Tensor with shape [C, H, W] and range [0, 1].
    Works without relying on torchvision to ensure it functions out of the box.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, C]
    arr = np.transpose(arr, (2, 0, 1))               # [C, H, W]
    return torch.from_numpy(arr)


def _domain_series_to_view(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Map the 'domain' column to a 0-based 'view' (scene1 -> 0, scene2 -> 1, …).
    If values are not in the form 'sceneN', assign stable indices by lexicographic
    order (0, 1, 2, ...).
    Returns: (view_series, mapping dict {domain_str: view_idx})
    """
    s = series.astype(str).str.lower().str.strip()
    pattern = re.compile(r"^scene(\d+)$")

    def parse_one(x: str) -> Optional[int]:
        m = pattern.match(x)
        if m:
            return int(m.group(1)) - 1  # 0-based
        return None

    parsed = s.map(parse_one)
    if parsed.notna().all():
        # Standard 'sceneN' form
        return parsed.astype(int), {f"scene{i}": i - 1 for i in sorted({int(re.match(r'^scene(\d+)$', v).group(1)) for v in s.unique()})}

    # Non-standard form: encode stably by lexicographic order
    uniq = sorted(s.unique())
    mapping = {name: i for i, name in enumerate(uniq)}
    view_series = s.map(mapping).astype(int)
    return view_series, mapping


class MultiSplitImageCSVDataset(Dataset):
    """
    Read datasets organized like:
      root/
        FALL/
          *.png
          FALL.csv
        REFRACTION/
          *.png
          REFRACTION.csv
        SCENE1/
          *.png
          SCENE1.csv
        ...
    The CSV must contain at least:
      - render_path (relative path)
      - 'view' or 'domain' (if 'domain' like 'scene1', it will be normalized to view=0, etc.)

    Other columns (all columns except 'id', 'render_path', 'domain', and the final 'view')
    are treated as rendering parameters and concatenated (in column order) into a float32
    metadata tensor (usable as supervision/regression targets). The length depends on the split.

    __getitem__ returns: (image_tensor, meta_tensor)
      - image_tensor: [C,H,W] float32, [0,1]
      - meta_tensor : [D] float32, containing 'view' and the rest parameter columns of this split

    Notes:
      * If images have varying sizes, you can use the collate function `collate_pad_to_max`
        provided in this file to automatically pad them in the DataLoader.
    """

    def __init__(
        self,
        root: str,
        split: str,
        image_transform: Optional[Any] = None,
        meta_float_dtype: torch.dtype = torch.float32,
        image_mode: str = "RGB",
        check_files: bool = False,
        auto_encode_non_numeric: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Args:
          root:    Dataset root directory
          split:   Specific split name (e.g., 'FALL', 'SCENE1', ...)
          image_transform: Optional image transform (e.g., torchvision.transforms.Compose([...]))
          meta_float_dtype: Dtype for the metadata tensor (default float32)
          image_mode: Image mode to open with ('RGB'/'L', etc.), default 'RGB'
          check_files: Whether to check and filter out missing image files at init (default False for faster init)
          auto_encode_non_numeric: Whether to automatically apply stable label encoding for non-numeric columns (default True)
          verbose: Whether to print a brief summary for this split at init
        """
        super().__init__()
        self.root = os.path.abspath(root)
        self.split = split
        self.image_transform = image_transform
        self.meta_float_dtype = meta_float_dtype
        self.image_mode = image_mode
        self.check_files = check_files
        self.auto_encode_non_numeric = auto_encode_non_numeric

        # ------- 1) Read CSV -------
        # Prefer CSV under the split directory; fallback to root/<split>.csv
        cand1 = os.path.join(self.root, split, f"{split}.csv")
        cand2 = os.path.join(self.root, f"{split}.csv")
        if os.path.isfile(cand1):
            csv_path = cand1
        elif os.path.isfile(cand2):
            csv_path = cand2
        else:
            raise FileNotFoundError(f"找不到 CSV：'{cand1}' 或 '{cand2}'")

        df = pd.read_csv(csv_path)
        expected_cols = set(df.columns)

        if "render_path" not in expected_cols:
            raise ValueError(f"{csv_path} need 'render_path'")

        # ------- 2) Normalize 'view' column -------
        self.view_mapping: Dict[str, int] = {}
        if "view" in df.columns:
            # Try to convert to int (allow string '0')
            df["view"] = pd.to_numeric(df["view"], errors="coerce").astype("Int64")
            if df["view"].isna().any():
                raise ValueError("列 'view' 存在无法转为数值的项。请检查 CSV。")
            df["view"] = df["view"].astype(int)
            if "domain" in df.columns:
                # Keep 'view', drop 'domain'
                df = df.drop(columns=["domain"])
        elif "domain" in df.columns:
            df["view"], self.view_mapping = _domain_series_to_view(df["domain"])
            df = df.drop(columns=["domain"])
        else:
            raise ValueError(f"{csv_path} 必须包含 'view' 或 'domain' 列")

        # ------- 3) Drop unused columns and determine parameter column order -------
        drop_cols = ["id", "render_path"]
        meta_cols = [c for c in df.columns if c not in drop_cols]

        # Treat all remaining CSV columns as parameters (including normalized 'view')
        # Automatically handle non-numeric columns with stable label encoding (optional)
        self.label_encoders: Dict[str, Dict[Any, float]] = {}
        for c in meta_cols:
            if c == "view":
                # Already integer; will be converted to float below
                continue
            # If non-numeric and needs encoding
            if not pd.api.types.is_numeric_dtype(df[c]) and self.auto_encode_non_numeric:
                uniq = sorted(df[c].astype(str).unique())  # stable sort
                enc = {v: float(i) for i, v in enumerate(uniq)}
                self.label_encoders[c] = enc
                df[c] = df[c].astype(str).map(enc)

        # Convert to float32
        for c in meta_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[meta_cols].isna().any().any():
            bad_cols = [c for c in meta_cols if df[c].isna().any()]
            raise ValueError(f"以下列存在无法转换为数值的数据：{bad_cols}。"
                             f"你可以关闭 auto_encode_non_numeric=False 并自行清洗，或检查 CSV。")

        # ------- 4) Optional: check files exist and filter -------
        if self.check_files:
            exists_mask = df["render_path"].apply(
                lambda p: os.path.isabs(p) and os.path.isfile(p) or os.path.isfile(os.path.join(self.root, p))
            )
            missing = (~exists_mask).sum()
            if missing > 0:
                print(f"[警告] {missing} 张图片在文件系统中不存在，已自动过滤。")
                df = df.loc[exists_mask].reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        self.meta_cols = meta_cols  # parameter columns including 'view'
        self.csv_path = csv_path

        # ------- 5) Some statistics -------
        # View counts
        view_counts = self.df["view"].value_counts().sort_index()
        self.view_counts = {int(k): int(v) for k, v in view_counts.items()}

        if verbose:
            print(self.summary())

    # ----------- Public attributes/methods -----------

    def summary(self) -> str:
        lines = [
            f"[MultiSplitImageCSVDataset] split = {self.split}",
            f"  root        : {self.root}",
            f"  csv         : {self.csv_path}",
            f"  num_samples : {len(self)}",
            f"  meta_cols   : {self.meta_cols}",
            f"  num_features: {self.num_features}",
            f"  view_counts : {self.view_counts}",
        ]
        if self.view_mapping:
            lines.append(f"  domain→view : {self.view_mapping}  (scene1→0 等价)")
        if self.label_encoders:
            lines.append(f"  encoded_cols: {list(self.label_encoders.keys())} (做了稳定标签编码)")
        return "\n".join(lines)

    @property
    def num_features(self) -> int:
        """Dimension of the metadata tensor (including 'view')."""
        return len(self.meta_cols)

    @property
    def meta_columns(self) -> List[str]:
        """Metadata column names (order equals tensor order)."""
        return list(self.meta_cols)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, rel_or_abs: str) -> str:
        # If render_path is absolute, return as-is; otherwise resolve relative to root
        if os.path.isabs(rel_or_abs):
            return rel_or_abs
        return os.path.join(self.root, rel_or_abs)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        row = self.df.iloc[index]
        img_path = self._resolve_image_path(row["render_path"])

        with Image.open(img_path) as im:
            if im.mode != self.image_mode:
                im = im.convert(self.image_mode)
            if self.image_transform is not None:
                img = self.image_transform(im)  # User-defined transform (e.g., Resize/Normalize/ToTensor)
                # Fallback if user transform didn't convert to Tensor
                if not isinstance(img, torch.Tensor):
                    img = _safe_pil_to_tensor(img)
            else:
                img = _safe_pil_to_tensor(im)

        # Build metadata tensor (column order equals self.meta_cols)
        meta = torch.as_tensor(row[self.meta_cols].to_numpy(dtype=np.float32), dtype=self.meta_float_dtype)
        return img, meta


# ------------------ Optional: padding collate for variable-size images ------------------

def collate_pad_to_max(batch: List[Tuple[Tensor, Tensor]], pad_value: float = 0.0) -> Tuple[Tensor, Tensor]:
    """
    Right-bottom pad images in the batch to the max H/W within the batch so they can
    be stacked to [B, C, H_max, W_max]. Metadata tensors are stacked as [B, D].
    """
    imgs, metas = zip(*batch)
    # Metadata tensors must share the same shape (single split satisfies this)
    metas = torch.stack(metas, dim=0)

    # Find max H/W in this batch
    C = imgs[0].shape[0]
    H_max = max(t.shape[1] for t in imgs)
    W_max = max(t.shape[2] for t in imgs)
    out = imgs[0].new_full((len(imgs), C, H_max, W_max), fill_value=pad_value)

    for i, t in enumerate(imgs):
        _, h, w = t.shape
        out[i, :, :h, :w] = t
    return out, metas


# ------------------ Convenience function to build a DataLoader ------------------

def build_dataloader(
    root: str,
    split: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pad_images: bool = False,
    **dataset_kwargs
) -> Tuple[DataLoader, MultiSplitImageCSVDataset]:
    """
    One-liner to create both DataLoader and Dataset.
      - When pad_images=True, use collate_pad_to_max to accommodate different image
        sizes within the same batch.
    Other arguments are passed through **dataset_kwargs to MultiSplitImageCSVDataset.
    """
    ds = MultiSplitImageCSVDataset(root=root, split=split, **dataset_kwargs)
    collate_fn = collate_pad_to_max if pad_images else None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return loader, ds
