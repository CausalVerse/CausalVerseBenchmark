import os
import random
import math
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.transforms as TT
from PIL import Image
import av
import decord
from decord import VideoReader
import threading
# from torchvision import transforms
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


class BaseDataset(ABC, Dataset):
    """
    Abstract base dataset class, all subdatasets should inherit from this class
    """
    def __init__(self, root_dir: str, transform=None):
        """
        Initialize base dataset
        
        Args:
            root_dir: Dataset root directory
            transform: Data transformation function
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = []
        self.dataset_info = {}
        
    @abstractmethod
    def load_data(self) -> None:
        """Abstract method to load data, must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Abstract method to get a single data item, must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Abstract method to return dataset size, must be implemented by subclasses"""
        pass
    
    def get_dataset_info(self) -> Dict:
        return self.dataset_info
    
    def get_statistics(self) -> Dict:
        return {
            "size": len(self),
            "root_directory": self.root_dir,
            "type": self.__class__.__name__
        }


class VideoDataset(BaseDataset):
    """Video dataset class with enhanced video processing capabilities"""
    
    def __init__(self, 
                 root_dir: str, 
                 transform=None, 
                 split=None,
                 image_size: Tuple[int, int] = (224, 224),
                 num_frames: int = 4,
                 sampling: str = "uniform",
                 skip_frames: float = 0.0,
                 clips_per_video: int = 1,
                 interval: int = 30,
                 max_start_frame: int = 2):
        """
        Initialize video dataset with advanced processing options
        
        Args:
            root_dir: Video data root directory
            transform: Video frame transformation function
            split: Dataset split (train/val/test) or None for all data
            image_size: Target frame size as (height, width)
            num_frames: Number of frames to extract from each video
            sampling: Frame sampling strategy ("uniform" or other strategies)
            skip_frames: Number of frames to skip at the beginning and end of videos
            clips_per_video: Number of clips to sample from each video
            interval: Frame interval between consecutive sampled frames
        """
        self.image_size = image_size
        self.num_frames = num_frames
        self.sampling = sampling
        self.skip_frames = skip_frames
        self.split = split
        self.clips_per_video = clips_per_video
        self.interval = interval
        self.max_start_frame = max_start_frame
        super().__init__(root_dir, transform)
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
                ]
            )
        self.load_data()
        
    def load_data(self) -> None:
        """Load video data with optional split filtering"""
        self.data_list = []
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        # Handle split if specified
        split_file = None
        if self.split:
            split_path = os.path.join(self.root_dir, f"{self.split}.txt")
            if os.path.exists(split_path):
                with open(split_path, 'r') as f:
                    split_file = set(line.strip() for line in f)
        
        # Walk through directory
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    # Check if file matches split requirements
                    if split_file is not None:
                        # Check if filename or relative path is in split file
                        rel_path = os.path.relpath(os.path.join(root, file), self.root_dir)
                        if file not in split_file and rel_path not in split_file:
                            continue
                    
                    video_path = os.path.join(root, file)
                    # Get video meta
                    video_meta = self._get_video_metadata(video_path)
                    if video_meta:
                        # Create an entry for each clip we want to sample
                        for clip_idx in range(self.clips_per_video):
                            self.data_list.append({
                                "path": video_path,
                                "clip_idx": clip_idx,
                                **video_meta
                            })
        
        self.dataset_info = {
            "num_videos": len(self.data_list) // self.clips_per_video,
            "total_clips": len(self.data_list),
            "clips_per_video": self.clips_per_video,
            "num_frames": self.num_frames,
            "interval": self.interval,
            "sampling": self.sampling,
            "image_size": self.image_size,
            "valid_extensions": valid_extensions,
            "split": self.split
        }
    
    def _get_video_metadata(self, video_path: str) -> Dict:
        """
        Extract metadata from a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video metadata or empty dict if extraction fails
        """
        try:
            with av.open(video_path, metadata_errors="ignore") as container:
                if not container.streams.video:
                    return {}
                
                # Get video stream
                video_stream = container.streams.video[0]
                fps = video_stream.average_rate
                duration = float(video_stream.duration * video_stream.time_base)
                width = video_stream.width
                height = video_stream.height
                nb_frames = video_stream.frames
                
                return {
                    "fps": float(fps) if fps else None,
                    "duration": duration,
                    "width": width,
                    "height": height,
                    "nb_frames": nb_frames if nb_frames > 0 else None
                }
        except Exception as e:
            # Log error but continue with dataset loading
            print(f"Error extracting metadata from {video_path}: {e}")
            return {}
            
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single video data item with processed frames
        
        Args:
            idx: Index of the video to retrieve
            
        Returns:
            Dictionary containing processed video frames and metadata
        """
        video_data = self.data_list[idx]
        video_path = video_data["path"]
        clip_idx = video_data["clip_idx"]
        
        try:
            # Process video with timeout protection
            frames, indices = self._process_video(
                video_path,
                image_size=self.image_size,
                duration=video_data.get("duration"),
                num_frames=self.num_frames,
                skip_frms_num=self.skip_frames,
                nb_read_frames=video_data.get("nb_frames"),
                clip_idx=clip_idx,
                interval=self.interval,
                max_start_frame=self.max_start_frame
            )
            
            if self.transform:
                frames = self.transform(frames)
                
            return {
                "frames": frames,
                "indices": indices,
                "path": video_path,
                "index": idx,
                "clip_idx": clip_idx,
                "video_meta": {
                    key: value for key, value in video_data.items() 
                    if key not in ["path", "clip_idx"]
                }
            }
        except Exception as e:
            # Return a placeholder on error
            print(f"Error processing video {video_path} (clip {clip_idx}): {e}")
            # Return black frames of the right shape
            frames = torch.zeros((self.num_frames, 3, *self.image_size), dtype=torch.uint8)
            return {
                "frames": frames,
                "indices": None,
                "path": video_path,
                "index": idx,
                "clip_idx": clip_idx,
                "video_meta": {
                    key: value for key, value in video_data.items() 
                    if key not in ["path", "clip_idx"]
                },
                "error": str(e)
            }
    
    def __len__(self) -> int:
        """Return number of video clips in dataset"""
        return len(self.data_list)
    
    def get_video_info(self, idx: int) -> Dict:
        """Get video info for a specific video"""
        return self.data_list[idx]
    
    def _process_video(self, 
                      video_path: str,
                      image_size: Optional[Tuple[int, int]] = None,
                      duration: Optional[float] = None,
                      num_frames: int = 4,
                      actual_fps: Optional[float] = None,
                      skip_frms_num: float = 0.0,
                      nb_read_frames: Optional[int] = None,
                      clip_idx: int = 0,
                      interval: int = 1,
                      max_start_frame: int = 1) -> torch.Tensor:
        """
        Process video to extract properly sized and sampled frames
        
        Args:
            video_path: Path to the video file
            image_size: Target frame size (height, width)
            duration: Video duration in seconds
            num_frames: Number of frames to extract
            actual_fps: Actual video FPS
            skip_frms_num: Number of frames to skip at start/end
            nb_read_frames: Total number of frames in video
            clip_idx: Index of the clip to sample (used for random start point)
            interval: Frame interval between consecutive sampled frames
            
        Returns:
            Tensor of processed video frames
        """
        video, indices = self._load_video_with_timeout(
            video_path,
            duration=duration,
            num_frames=num_frames,
            actual_fps=actual_fps,
            skip_frms_num=skip_frms_num,
            nb_read_frames=nb_read_frames,
            clip_idx=clip_idx,
            interval=interval,
            max_start_frame=max_start_frame
        )
        video = video.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
        if image_size is not None:
            video = self.resize_video_tensor(video, image_size)

        return video, indices
    
    def _load_video_with_timeout(self, *args, **kwargs):
        """
        Load video with timeout protection to prevent hanging
        
        Returns:
            Video tensor or raises TimeoutError
        """
        video_container = {}

        def target_function():
            video, indices = self._load_video(*args, **kwargs)
            video_container["video"] = video
            video_container["indices"] = indices

        thread = threading.Thread(target=target_function)
        thread.start()
        timeout = 20
        thread.join(timeout)

        if thread.is_alive():
            print("Loading video timed out")
            raise TimeoutError
        return video_container.get("video", None).contiguous(), video_container.get("indices", None)
    
    def _load_video(self,
                  video_path,
                  sampling="uniform",
                  duration=None,
                  num_frames=4,
                  skip_frms_num=0.0,
                  nb_read_frames=None,
                  actual_fps=60,
                  clip_idx=0,
                  interval=1,
                  max_start_frame=1):
        """
        Load video frames with the specified sampling strategy
        
        Args:
            video_path: Path to video
            sampling: Frame sampling strategy
            duration: Video duration
            num_frames: Number of frames to extract
            skip_frms_num: Frames to skip
            nb_read_frames: Total frames in video
            clip_idx: Index of the clip to sample (used for random start point)
            interval: Frame interval between consecutive sampled frames
            
        Returns:
            Tensor of sampled video frames
        """
        decord.bridge.set_bridge("torch")
        vr = VideoReader(uri=video_path, height=-1, width=-1)
        
        if nb_read_frames is not None:
            ori_vlen = nb_read_frames
        else:
            ori_vlen = min(int(duration * actual_fps) - 1, len(vr)) if duration and actual_fps else len(vr)
        
        
        total_frames_needed = (num_frames - 1) * interval + 1
        
        # Calculate start and end frame indices
        max_seek = max(0, int(ori_vlen - skip_frms_num - total_frames_needed))
        
        # Use clip_idx to generate a random start position if multiple clips
        if sampling == "uniform":
            start = random.randint(0, max_start_frame)
                
            end = min(start + total_frames_needed, len(vr))
            
            # Generate indices with specified interval
            if end - start < total_frames_needed:
                # Not enough frames to maintain interval, use linspace
                indices = np.linspace(start, max(start, end-1), num_frames).astype(int)
            else:
                indices = np.array([start + i * interval for i in range(num_frames)])
        else:
            raise NotImplementedError(f"Sampling strategy '{sampling}' not implemented")
        # Get frames
        try:
            temp_frms = vr.get_batch(indices)
            if temp_frms is None:
                raise ValueError("Failed to get frames from video")
                
            tensor_frms = torch.from_numpy(temp_frms) if not isinstance(temp_frms, torch.Tensor) else temp_frms
            
            # Pad if necessary
            return self._pad_last_frame(tensor_frms, num_frames), indices
            
        except Exception as e:
            print(f"Error in frame extraction: {e}")
            # Return black frames
            return torch.zeros((num_frames, vr.height, vr.width, 3), dtype=torch.uint8), None
    
    def _pad_last_frame(self, tensor, num_frames):
        """
        Pad tensor with copies of the last frame if needed
        
        Args:
            tensor: Video tensor of shape [T, H, W, C]
            num_frames: Target number of frames
            
        Returns:
            Padded tensor with exactly num_frames
        """
        # T, H, W, C
        if len(tensor) < num_frames:
            pad_length = num_frames - len(tensor)
            # Use the last frame to pad instead of zero
            last_frame = tensor[-1]
            pad_tensor = last_frame.unsqueeze(0).expand(pad_length, *tensor.shape[1:])
            padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
            return padded_tensor
        else:
            return tensor[:num_frames]
    
    def _resize_for_rectangle_crop(self, arr, image_size, reshape_mode="random"):
        """
        Resize video preserving aspect ratio then crop to target size
        
        Args:
            arr: Video tensor [T, C, H, W]
            image_size: Target size (H, W)
            reshape_mode: Cropping strategy ("random", "center", or "none")
            
        Returns:
            Resized and cropped video tensor
        """
        # First add batch dimension for resize function
        arr = arr.unsqueeze(0)
        
        if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
            # Width is relatively larger
            arr = TF.resize(
                arr,
                size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
                interpolation=InterpolationMode.BICUBIC,
            )
        else:
            # Height is relatively larger
            arr = TF.resize(
                arr,
                size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
                interpolation=InterpolationMode.BICUBIC,
            )

        h, w = arr.shape[2], arr.shape[3]
        arr = arr.squeeze(0)  # Remove batch dimension

        delta_h = h - image_size[0]
        delta_w = w - image_size[1]

        # Determine crop position
        if reshape_mode == "random":
            top = np.random.randint(0, delta_h + 1)
            left = np.random.randint(0, delta_w + 1)
        elif reshape_mode == "center":
            top, left = delta_h // 2, delta_w // 2
        else:
            raise NotImplementedError(f"Reshape mode '{reshape_mode}' not implemented")
            
        # Crop to target size
        arr = TT.functional.crop(arr, top=top, left=left, height=image_size[0], width=image_size[1])
        return arr

    def resize_video_tensor(self, arr: torch.Tensor,
                        image_size: tuple[int, int],
                        interpolation: InterpolationMode = InterpolationMode.BICUBIC) -> torch.Tensor:
        """
        Resize a video tensor [T, C, H, W] to target size (H, W), preserving aspect ratio and cropping center.

        Args:
            video: Input video tensor of shape [T, C, H, W]
            target_size: Target spatial size (height, width)
            interpolation: Interpolation mode (default: BICUBIC)

        Returns:
            Resized and cropped video tensor of shape [T, C, target_H, target_W]
        """
        T, C, H, W = arr.shape
        target_h, target_w = image_size
        resized_frames = []

        for t in range(T):
            frame = arr[t]  # [C, H, W]

            aspect_ratio_in = W / H
            aspect_ratio_out = target_w / target_h

            if aspect_ratio_in > aspect_ratio_out:
                new_h = target_h
                new_w = int(W * target_h / H)
            else:
                new_w = target_w
                new_h = int(H * target_w / W)

            resized = TF.resize(frame, size=[new_h, new_w], interpolation=interpolation)

            cropped = TF.center_crop(resized, output_size=[target_h, target_w])

            resized_frames.append(cropped)

        return torch.stack(resized_frames)  # [T, C, target_h, target_w]


def show_video_frames(frames: torch.Tensor, title: str = "Video Frames"):
    frames = frames.permute(0, 2, 3, 1).numpy()  # [T, C, H, W] -> [T, H, W, C]
    fig, axes = plt.subplots(1, len(frames), figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(frames[i].astype("uint8"))
        ax.set_title(f"Frame {i}")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig("test.png")

def main():
    root_dir = "/video"

    
    transform = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
        ]
    )

    dataset = VideoDataset(
        root_dir=root_dir,
        transform=transform,
        split=None,  
        image_size=(128, 128),
        num_frames=4,
        sampling="uniform",
        skip_frames=0.0,
        clips_per_video=1,
        interval=1
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(len(dataset))
    for batch in dataloader:
        frames = batch["frames"][0]  # shape: [T, C, H, W]
        video_path = batch["path"][0]
        print(f"Loaded video from: {video_path}")
        show_video_frames(frames, title=f"Video: {os.path.basename(video_path)}")
        break

if __name__ == "__main__":
    main()