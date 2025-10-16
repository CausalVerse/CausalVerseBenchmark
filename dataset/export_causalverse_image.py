# -*- coding: utf-8 -*-
"""
Export images and per-split CSVs from the Hugging Face dataset repo
`CausalVerse/CausalVerse_Image`, using local parquet files to avoid
UnexpectedSplitsError. The script:
- Downloads parquet files with a file-level progress bar (and byte-level
  progress provided by huggingface_hub).
- Discovers splits from local parquet filenames.
- Exports images to: image/<SPLIT_CASED>/xxx.png (case strategy is configurable).
- Builds CSV headers strictly by the first-seen order of keys in the "metavalue"
  field of that split; optionally appends a trailing "render_path" column.
- Can be used both as a CLI and as an importable module with default config.
"""

import os
import io
import re
import csv
import json
import ast
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Iterable, Optional, Tuple

from tqdm import tqdm
from PIL import Image
from datasets import load_dataset, Image as HFImage
from huggingface_hub import list_repo_files, hf_hub_download


# ============================ Default Config ============================

@dataclass
class ExportConfig:
    # Hugging Face dataset repo
    repo_id: str = "CausalVerse/CausalVerse_Image"

    # Local directories
    hf_home: Path = Path("./.hf")                      # HF cache in current directory
    raw_repo_dir: Path = Path("./CausalVerse_Image")   # Repo snapshot (actual files, incl. parquet)
    image_root: Path = Path("./image")                 # Image export root

    # Export behavior
    folder_case: str = "upper"                         # "upper" | "capitalize" | "keep"
    overwrite: bool = False                            # Overwrite existing images
    splits_to_export: Optional[List[str]] = None       # None = discover all; or e.g. ["FALL","SCENE1"] (case-insensitive)
    include_render_path_column: bool = True            # Append "render_path" to CSV rows

    # Download behavior
    download_allow_patterns: Optional[List[str]] = field(default_factory=lambda: ["data/*.parquet"])
    skip_download_if_local: bool = True                # If parquet exists locally, skip download


# ============================ Environment ============================

def setup_environment(cfg: ExportConfig) -> None:
    """
    Prepare environment variables and required directories.
    """
    os.environ["HF_HOME"] = str(cfg.hf_home.resolve())
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"     # Enable fast transfer if available
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # Show byte-level progress
    cfg.hf_home.mkdir(parents=True, exist_ok=True)
    cfg.image_root.mkdir(parents=True, exist_ok=True)


# ============================ Download ============================

def _matches_any(path: str, patterns: List[str]) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(path, p) for p in patterns)

def download_repo_with_progress(
    repo_id: str,
    local_dir: Path,
    allow_patterns: Optional[List[str]] = None,
    hf_home: Optional[Path] = None,
) -> None:
    """
    Copy files from the HF dataset repo into `local_dir`.
    Outer tqdm shows file-level progress; huggingface_hub shows byte-level progress.
    """
    print(f"Listing files for {repo_id} ...")
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    if allow_patterns:
        files = [f for f in files if _matches_any(f, allow_patterns)]
    print(f"Total files to download: {len(files)}")
    local_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = str((hf_home or Path(os.environ.get("HF_HOME", "./.hf"))).resolve())

    for f in tqdm(files, desc="Downloading repo files", unit="file"):
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f,
            cache_dir=cache_dir,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # copy real files (no symlinks)
        )


# ============================ Utilities ============================

def norm_split_dir(name: str, folder_case: str) -> str:
    """
    Normalize split directory name according to case strategy.
    """
    if folder_case == "upper":
        return name.upper()
    if folder_case == "capitalize":
        return name.capitalize()
    return name

def parse_metavalue(meta: Any) -> Dict[str, Any]:
    """
    Robustly parse 'metavalue' into a dict.
    Accepts JSON, Python literal (e.g., with single quotes), "k:v" or "k=v" text.
    Returns {} if cannot be parsed.
    """
    if isinstance(meta, dict):
        return meta
    s = "" if meta is None else str(meta).strip()
    if not s:
        return {}

    # 1) JSON
    try:
        v = json.loads(s)
        if isinstance(v, dict):
            return v
    except Exception:
        pass

    # 2) Python literal (e.g., single-quoted dict)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, dict):
            return v
    except Exception:
        pass

    # 3) "k:v" or "k=v"
    out: Dict[str, Any] = {}
    for k, v in re.findall(r"([A-Za-z0-9_]+)\s*[:=]\s*([^,;]+)", s):
        out[k.strip()] = v.strip()
    return out

def coerce_value(v: Any) -> Any:
    """
    Best-effort coercion from string to number/None; otherwise return as-is.
    """
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in {"", "none", "null", "nan"}:
        return None
    # int
    if re.fullmatch(r"[-+]?\d+", s):
        try:
            return int(s)
        except Exception:
            return s
    # float
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", s):
        try:
            return float(s)
        except Exception:
            return s
    return v

def ensure_csv_header(csv_path: Path, header: Iterable[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(list(header))

def save_image_any(img_obj: Any, out_path: Path) -> None:
    """
    Save image to `out_path`. Supports:
      - PIL.Image.Image (recommended: via datasets.Image(decode=True)),
      - dict({'bytes'|'path'}), bytes/bytearray/memoryview,
      - objects with .to_pybytes(), or a local path string.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Already a PIL Image
    if isinstance(img_obj, Image.Image):
        img = img_obj
    else:
        # 2) Try to obtain raw bytes, then open with PIL
        data = None
        if isinstance(img_obj, dict):
            data = img_obj.get("bytes")
            p = img_obj.get("path")
            if data is None and p:
                with open(p, "rb") as rf:
                    data = rf.read()
        elif isinstance(img_obj, (bytes, bytearray, memoryview)):
            data = bytes(img_obj)
        elif hasattr(img_obj, "to_pybytes"):  # e.g., pyarrow.Buffer
            data = img_obj.to_pybytes()
        elif isinstance(img_obj, str) and os.path.exists(img_obj):
            with open(img_obj, "rb") as rf:
                data = rf.read()

        if data is None:
            raise ValueError(
                f"Unsupported image object: {type(img_obj)} "
                f"(ensure cast_column('image', datasets.Image(decode=True)))"
            )

        img = Image.open(io.BytesIO(data))

    if not out_path.suffix:
        out_path = out_path.with_suffix(".png")
    img.save(out_path)

def unique_name(dir_path: Path, filename: str, overwrite: bool) -> str:
    """
    Avoid overwriting: if file exists and overwrite=False, append _1, _2, ...
    """
    stem, ext = os.path.splitext(filename)
    cand = filename
    i = 1
    while (dir_path / cand).exists() and not overwrite:
        cand = f"{stem}_{i}{ext}"
        i += 1
    return cand


# ============================ Parquet & Splits ============================

def find_parquet_files(parquet_dir: Path) -> Dict[str, List[str]]:
    """
    Find parquet files under `parquet_dir` and map split -> list of parquet paths.
    Expects filenames like: <split>-00000-of-00005.parquet
    """
    all_parquets = sorted(glob.glob(str(parquet_dir / "*.parquet")))
    split_to_files: Dict[str, List[str]] = {}
    pat = re.compile(r"^([A-Za-z0-9_]+)-\d{5}-of-\d{5}\.parquet$")

    for p in all_parquets:
        name = Path(p).name
        m = pat.match(name)
        if m:
            split = m.group(1)  # e.g., fall, scene1
            split_to_files.setdefault(split, []).append(p)
    return split_to_files

def resolve_splits(split_to_files: Dict[str, List[str]], requested: Optional[List[str]]) -> List[str]:
    """
    Determine which splits to export: all discovered or a case-insensitive subset.
    """
    if not split_to_files:
        raise RuntimeError("No parquet files found under the expected 'data/' directory.")

    discovered = sorted(split_to_files.keys())
    if requested is None:
        splits = discovered
    else:
        low = {s.lower(): s for s in discovered}
        splits: List[str] = []
        for s in requested:
            k = s.lower()
            if k in low:
                splits.append(low[k])
            else:
                print(f"[warn] Split {s} not found locally, skip.")
    print("Splits to export (from local parquet):", splits)
    return splits


# ============================ Export Core ============================

def scan_metavalue_keys(files: List[str], split_name: str) -> List[str]:
    """
    Iterate once over the split to collect metavalue keys in first-seen order.
    Force 'image' decode to PIL to standardize the export path later.
    """
    ds_scan = load_dataset("parquet", data_files=files, split="train")
    if "image" in ds_scan.features:
        ds_scan = ds_scan.cast_column("image", HFImage(decode=True))

    header_keys: List[str] = []
    for ex in tqdm(ds_scan, total=len(ds_scan), desc=f"[1/2] scan metavalue keys: {split_name}"):
        meta = parse_metavalue(ex.get("metavalue"))
        for k in meta.keys():
            if k not in header_keys:
                header_keys.append(k)
    return header_keys

def export_split(
    split_name: str,
    files: List[str],
    cfg: ExportConfig,
) -> Tuple[int, int]:
    """
    Export one split:
    - Build CSV header strictly by first-seen metavalue key order (for this split).
    - Export images.
    - Write CSV rows with coerced values and optionally a trailing 'render_path'.
    Returns (ok, failed).
    """
    header_keys = scan_metavalue_keys(files, split_name)
    header = list(header_keys) + (["render_path"] if cfg.include_render_path_column else [])

    split_dirname = norm_split_dir(split_name, cfg.folder_case)
    out_dir = cfg.image_root / split_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{split_dirname}.csv"
    ensure_csv_header(out_csv, header)

    # Reload to iterate from the beginning; cast image to PIL
    ds = load_dataset("parquet", data_files=files, split="train")
    if "image" in ds.features:
        ds = ds.cast_column("image", HFImage(decode=True))

    fail, ok = 0, 0
    with out_csv.open("a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        for ex in tqdm(ds, total=len(ds), desc=f"[2/2] export images & CSV: {split_name}"):
            rp = str(ex.get("render_path") or "")
            base = os.path.basename(rp) if rp else None
            if not base:
                base = f"{ok+fail}.png"  # Fallback filename
            dest_name = unique_name(out_dir, base, cfg.overwrite)
            dest_path = out_dir / dest_name

            # Save image
            try:
                save_image_any(ex["image"], dest_path)
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[warn] save image failed: {dest_path} ({e})")

            # Write CSV row: only metavalue keys; optionally add relative render_path
            meta = parse_metavalue(ex.get("metavalue")) or {}
            row = [coerce_value(meta.get(k)) for k in header_keys]
            if cfg.include_render_path_column:
                row.append(f"{split_dirname}/{dest_name}")
            writer.writerow(row)

    print(f"Done {split_name}: ok={ok}, failed={fail}")
    return ok, fail


# ============================ Driver ============================

def maybe_download_repo(cfg: ExportConfig) -> None:
    """
    Download parquet files if necessary, with progress.
    """
    parquet_dir = cfg.raw_repo_dir / "data"
    need_download = True
    if cfg.skip_download_if_local and parquet_dir.exists():
        if glob.glob(str((parquet_dir / "*.parquet").resolve())):
            need_download = False

    if need_download:
        download_repo_with_progress(
            repo_id=cfg.repo_id,
            local_dir=cfg.raw_repo_dir,
            allow_patterns=cfg.download_allow_patterns,
            hf_home=cfg.hf_home,
        )
        print("Repo snapshot prepared.\n")
    else:
        print("Local parquet found. Skip downloading.\n")

def export_all(cfg: ExportConfig) -> None:
    """
    Orchestrate the full export across splits.
    """
    setup_environment(cfg)
    maybe_download_repo(cfg)

    parquet_dir = cfg.raw_repo_dir / "data"
    split_to_files = find_parquet_files(parquet_dir)
    splits = resolve_splits(split_to_files, cfg.splits_to_export)

    for sp in splits:
        print(f"\n>>> Exporting split: {sp}")
        files = split_to_files[sp]
        export_split(sp, files, cfg)

    print("\nAll done. Images are under ./image/<SPLIT>/, and CSVs are next to them.")


# ============================ CLI ============================

def build_arg_parser(defaults: ExportConfig):
    import argparse

    p = argparse.ArgumentParser(
        description="Export CausalVerse_Image dataset to local images + per-split CSVs (from local parquet)."
    )

    # Repo & paths
    p.add_argument("--repo-id", default=defaults.repo_id, help="HF dataset repo id.")
    p.add_argument("--hf-home", type=Path, default=defaults.hf_home, help="HF cache directory.")
    p.add_argument("--raw-repo-dir", type=Path, default=defaults.raw_repo_dir, help="Local repo snapshot directory.")
    p.add_argument("--image-root", type=Path, default=defaults.image_root, help="Image export root directory.")

    # Export behavior
    p.add_argument("--folder-case", choices=["upper", "capitalize", "keep"],
                   default=defaults.folder_case, help="Directory name case strategy for splits.")
    p.add_argument("--overwrite", action="store_true", default=defaults.overwrite,
                   help="Overwrite existing images (default: False).")
    p.add_argument("--no-overwrite", action="store_false", dest="overwrite",
                   help="Do not overwrite existing images.")
    p.add_argument("--splits", nargs="+", default=None,
                   help="Specific splits to export (case-insensitive). If omitted, export all discovered splits.")
    p.add_argument("--include-render-path-column", action="store_true",
                   default=defaults.include_render_path_column,
                   help="Append 'render_path' column to CSV.")
    p.add_argument("--no-include-render-path-column", action="store_false",
                   dest="include_render_path_column",
                   help="Do not append 'render_path' column to CSV.")

    # Download behavior
    p.add_argument("--download-allow-patterns", nargs="+",
                   default=defaults.download_allow_patterns,
                   help="Glob patterns to download from the repo (space-separated).")
    p.add_argument("--skip-download-if-local", action="store_true",
                   default=defaults.skip_download_if_local,
                   help="Skip download if local parquet files exist.")
    p.add_argument("--no-skip-download-if-local", action="store_false",
                   dest="skip_download_if_local",
                   help="Force download even if local parquet files exist.")

    return p

def main():
    defaults = ExportConfig()
    parser = build_arg_parser(defaults)
    args = parser.parse_args()

    cfg = ExportConfig(
        repo_id=args.repo_id,
        hf_home=args.hf_home,
        raw_repo_dir=args.raw_repo_dir,
        image_root=args.image_root,
        folder_case=args.folder_case,
        overwrite=args.overwrite,
        splits_to_export=args.splits,
        include_render_path_column=args.include_render_path_column,
        download_allow_patterns=args.download_allow_patterns,
        skip_download_if_local=args.skip_download_if_local,
    )

    export_all(cfg)

if __name__ == "__main__":
    main()
