import os
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import torch
from torch.utils.data import Dataset, ConcatDataset
import sys
sys.path.append('../')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data_modules import PhysicsFallDataset, PhysicsProjectileDataset, PhysicsCollisionDataset, FixedRoboticsDataset
from data_modules.video_dataset import VideoDataset, BaseDataset

class DataFactory:
    """
    Data factory class responsible for creating and managing datasets
    This factory handles creating datasets with specific configurations and keeps track of them
    """
    
    def __init__(self):
        """Initialize the data factory with an empty dataset registry"""
        self.datasets = {}
        self.configs = {}
    
    def create_dataset(self, name: str, dataset_type: str, dataset_path: str, split: Optional[str] = None, **kwargs) -> Dataset:
        """
        Create a dataset of the specified type with the given parameters
        
        Args:
            name: Unique identifier for the dataset
            dataset_type: Type of dataset to create ("video", "image", etc.)
            dataset_path: Root directory containing the dataset
            split: Dataset split to use (train/val/test) or None for all data
            **kwargs: Additional arguments specific to the dataset type
            
        Returns:
            The created dataset instance
            
        Raises:
            ValueError: If the dataset type is not supported or the dataset path doesn't exist
        """
        # Check if dataset path exists
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path '{dataset_path}' does not exist")
            
        # Create dataset based on type
        if dataset_type.lower() == "video":
            dataset = self._create_video_dataset(name, dataset_path, split, **kwargs)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
        # Store dataset and its configuration
        self.datasets[name] = dataset
        self.configs[name] = {
            "type": dataset_type,
            "path": dataset_path,
            "split": split,
            **kwargs
        }
        
        return dataset
    
    def _create_video_dataset(self, name: str, dataset_path: str, split: Optional[str] = None, **kwargs) -> VideoDataset:
        """
        Create a video dataset with the given parameters
        
        Args:
            dataset_path: Root directory containing the video dataset
            split: Dataset split to use (train/val/test) or None for all data
            **kwargs: Additional arguments for VideoDataset
            
        Returns:
            Configured VideoDataset instance
        """
        if name == "collision" or name == "collision_simple" or name == "collision_complex":
            return PhysicsCollisionDataset(
                root_dir=dataset_path,
                **kwargs
            )
        elif name == "projectile" or name == "projectile_simple" or name == "projectile_complex":
            return PhysicsProjectileDataset(
                root_dir=dataset_path,
                **kwargs
            )
        elif name == "fall" or name == "fall_simple" or name == "fall_complex":
            return PhysicsFallDataset(
                root_dir=dataset_path,
                **kwargs
            )
        elif name == "fixed_robotics" or name == "fixed_robotics_kitchen" or name == "fixed_robotics_study":
            return FixedRoboticsDataset(
                root_dir=dataset_path,
                **kwargs
            )
        else:
            return VideoDataset(
                root_dir=dataset_path,
                split=split,
                **kwargs
                )
    
    def get_dataset(self, name: str) -> Optional[Dataset]:
        """
        Retrieve a dataset by name
        
        Args:
            name: Name of the dataset to retrieve
            
        Returns:
            The dataset if found, None otherwise
        """
        return self.datasets.get(name)
    
    def create_combined_dataset(self, name: str, dataset_names: List[str]) -> Dataset:
        """
        Create a combined dataset from multiple existing datasets
        
        Args:
            name: Name for the new combined dataset
            dataset_names: List of dataset names to combine
            
        Returns:
            Combined dataset
            
        Raises:
            ValueError: If any of the specified datasets don't exist
        """
        datasets_to_combine = []
        for dataset_name in dataset_names:
            dataset = self.get_dataset(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset '{dataset_name}' not found")
            datasets_to_combine.append(dataset)
            
        combined_dataset = ConcatDataset(datasets_to_combine)
        self.datasets[name] = combined_dataset
        
        return combined_dataset
    
    def list_datasets(self) -> List[str]:
        """
        List all available datasets
        
        Returns:
            List of dataset names
        """
        return list(self.datasets.keys())
    
    def get_dataset_info(self, name: str) -> Dict:
        """
        Get information about a specific dataset
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dictionary containing dataset information
        """
        dataset = self.get_dataset(name)
        if dataset is None:
            return {}
            
        config = self.configs.get(name, {})
    
            
        if hasattr(dataset, 'get_statistics') and callable(dataset.get_statistics):
            statistics = dataset.get_statistics()
        else:
            statistics = {"size": len(dataset)}
            
        return {
            "name": name,
            "type": config.get("type", dataset.__class__.__name__),
            "size": len(dataset),
            "config": config,
            "statistics": statistics
        }


# Usage example
if __name__ == "__main__":
    # Create data factory
    factory = DataFactory()
    
    # Create a video dataset
    video_dataset = factory.create_dataset(
        name="fall",
        dataset_type="video",
        dataset_path="dataset/physical_simulation/free_fall_simple",
        split="",
        image_size=(256, 256),
        num_frames=16,
        sampling="uniform",
        only_latents=True,
    )

    print(f"Available datasets: {factory.list_datasets()}")
    
    print(f"Dataset info: {factory.get_dataset_info('fall_img')}")
    
    
    # Get a single sample from the dataset
    if len(video_dataset) > 0:
        sample = video_dataset[0]
        print(f"Dataset Size: {len(video_dataset)}")
        print(f"Sample keys: {sample.keys()}")
        print(f"Sample latent shape: {sample['latent'].shape if 'latent' in sample else 'N/A'}")
        print(f"Sample vector shape: {sample['vector'].shape if 'vector' in sample else 'N/A'}")
        print(f"Sample frame indices: {sample['frame_indices'] if 'frame_indices' in sample else 'N/A'}")
