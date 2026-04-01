import os
import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_modules.video_dataset import VideoDataset

class PhysicsFallDataset(VideoDataset):
    """
    Dataset for physics collision videos with multiple camera views and associated metadata
    """
    def __init__(self, 
                 root_dir: str, 
                 transform=None, 
                 split=None,
                 image_size: Tuple[int, int] = (224, 224),
                 num_frames: int = 8,
                 sampling: str = "uniform",
                 skip_frames: float = 0.0,
                 clips_per_video: int = 1,
                 interval: int = 30,
                 max_start_frame: int = 2,
                 camera_views: List[str] = None,
                 include_depth: bool = False,
                 only_latents: bool = False,
                 latent_view: str = "front"):
        """
        Initialize physics collision dataset
        
        Args:
            root_dir: Dataset root directory containing meta and video subdirectories
            transform: Video frame transformation function
            split: Dataset split or None for all data
            image_size: Target frame size as (height, width)
            num_frames: Number of frames to extract from each video
            sampling: Frame sampling strategy
            skip_frames: Number of frames to skip at the beginning and end
            camera_views: List of camera views to include (if None, uses all available)
            include_depth: Whether to include depth videos in addition to color videos
        """
        self.camera_views = camera_views if camera_views else ["bird", "front", "left", "right"]
        self.include_depth = include_depth
        self.only_latents = only_latents
        self.latent_view = latent_view
        
        self.meta_dir = os.path.join(root_dir, "meta")
        self.video_dir = os.path.join(root_dir, "video")
        self.latents_dir = os.path.join(root_dir, "latents", latent_view)
        self.uuid_to_meta = {}  # Maps UUID to metadata

        super().__init__(root_dir, transform, split, image_size,
                         num_frames, sampling, skip_frames, clips_per_video,
                         interval, max_start_frame)

    def load_data(self) -> None:
        """Load video data and associated metadata"""
        self.data_list = []
        
        if not os.path.exists(self.meta_dir):
            print(f"Error: Meta directory not found in {self.root_dir}")
            return
        
        if self.only_latents:
            if not os.path.exists(self.latents_dir):
                print(f"Error: Latents directory not found in {self.root_dir}")
                return
        else:
            if not os.path.exists(self.video_dir):
                print(f"Error: Video directory not found in {self.root_dir}")
                return
        
        json_files = [f for f in os.listdir(self.meta_dir) if f.endswith('.json')]
        
        for json_file in json_files:
            json_stem = json_file.split('.')[0]
            uuid = json_stem[:-5] if json_stem.endswith('_meta') else json_stem
            json_path = os.path.join(self.meta_dir, json_file)
            npz_path = os.path.join(self.meta_dir, f"{uuid}.npz")
            
            if not os.path.exists(npz_path):
                print(f"Warning: NPZ file not found for {uuid}")
                continue
            
            try:
                with open(json_path, 'r') as f:
                    sample_metadata = json.load(f)
            except Exception as e:
                print(f"Error loading JSON for {uuid}: {e}")
                continue
            
            self.uuid_to_meta[uuid] = sample_metadata
            if self.only_latents:
                for sample_idx in range(self.clips_per_video):
                    if self.clips_per_video == 1:
                        latent_filename = f"{uuid}.pt"
                    else:
                        latent_filename = f"{uuid}_{sample_idx}.pt"
                    
                    latent_path = os.path.join(self.latents_dir, latent_filename)
                    
                    if os.path.exists(latent_path):
                        sample_data = {
                            "uuid": uuid,
                            "sample_idx": sample_idx,
                            "latent_path": latent_path,
                            "npz_path": npz_path,
                            "json_path": json_path,
                        }
                        self.data_list.append(sample_data)
            else:
                for view in self.camera_views:
                    color_video_path = os.path.join(self.video_dir, f"{uuid}_{view}_color.mp4")
                    # print(f"color_video_path: {color_video_path}")
                    if not os.path.exists(color_video_path):
                        continue
                    
                    video_metadata = self._get_video_metadata(color_video_path)
                    if not video_metadata:
                        continue
                    
                    depth_video_path = None
                    if self.include_depth:
                        depth_video_path = os.path.join(self.video_dir, f"{uuid}_{view}_depth.mp4")
                        if not os.path.exists(depth_video_path):
                            depth_video_path = None
                    
                    for clip_idx in range(self.clips_per_video):
                        # Create a unique sample entry for each clip
                        sample_data = {
                            "uuid": uuid,
                            "camera_view": view,
                            "clip_idx": clip_idx,
                            "color_path": color_video_path,
                            "depth_path": depth_video_path,
                            "npz_path": npz_path,
                            "json_path": json_path,
                            **video_metadata  
                        }
                        self.data_list.append(sample_data)
        
        self.dataset_info.update({
            "num_unique_samples": len(self.uuid_to_meta),
            "camera_views": self.camera_views,
            "latent_view": self.latent_view,
            "include_depth": self.include_depth,
            "only_latents": self.only_latents
        })
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a video sample with associated metadata
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing video frames and associated metadata
        """
        sample_data = self.data_list[idx]
        uuid = sample_data["uuid"]
        if self.only_latents:
            try:
                latent_data = torch.load(sample_data["latent_path"])
                mean = latent_data["latents"]  
                frame_indices = latent_data["frame_numbers"]  # list of frame indices
                
                json_metadata = self.uuid_to_meta[uuid]
                npz_data = self._load_npz_data(sample_data["npz_path"], json_metadata, np.array(frame_indices))
                
                global_vec = self.encode_global_to_vector_simple(json_metadata.get("Global", {}))
                position = npz_data["position"][0]  
                rotation = npz_data["rotation"][0]  
               
                vector = self.merge_content_and_style_as_sequence(global_vec, position, rotation)
                
                result = {
                    "latent": mean,                    
                    "vector": vector,               
                    "frame_indices": frame_indices  
                }
                
                return result
                
            except Exception as e:
                print(f"Error processing latents for sample {uuid}: {e}")
                
                error_result = {
                    "mean": torch.zeros((16, 4, 64, 64)),
                    "logvar": torch.zeros((16, 4, 64, 64)),
                    "vector": torch.zeros((16, 6)),  
                    "frame_indices": list(range(16))
                }
                return error_result 
        
        else:
            clip_idx = sample_data.get("clip_idx", 0)
            camera_view = sample_data["camera_view"]
            try:
                color_frames, indices = self._process_video(
                    sample_data["color_path"],
                    image_size=self.image_size,
                    duration=sample_data.get("duration"),
                    num_frames=self.num_frames,
                    skip_frms_num=self.skip_frames,
                    nb_read_frames=sample_data.get("nb_frames"),
                    clip_idx=clip_idx,
                    interval=self.interval,
                    max_start_frame=self.max_start_frame
                )

                if self.transform:
                    color_frames = self.transform(color_frames)
                
                depth_frames = None
                if sample_data["depth_path"]:
                    depth_frames = self._process_video(
                        sample_data["depth_path"],
                        image_size=self.image_size,
                        duration=sample_data.get("duration"),
                        num_frames=self.num_frames,
                        skip_frms_num=self.skip_frames,
                        nb_read_frames=sample_data.get("nb_frames"),
                        clip_idx=clip_idx,
                        interval=self.interval,
                        max_start_frame=self.max_start_frame
                    )
                    
                    if self.transform:
                        depth_frames = self.transform(depth_frames)
                json_metadata = self.uuid_to_meta[uuid]
                npz_data = self._load_npz_data(sample_data["npz_path"], json_metadata, indices)
                
                global_vec = self.encode_global_to_vector_simple(json_metadata.get("Global", {}))
                position = npz_data["position"][0]  # [T, D_p]
                rotation = npz_data["rotation"][0]  # [T, D_r]

                vector = self.merge_content_and_style_as_sequence(global_vec, position, rotation)
               
                color_frames = color_frames.permute(1, 0, 2, 3) 

                result = {
                    "frames": color_frames,                  
                    "vector": vector,                       
                }
                if depth_frames is not None:
                    result["depth_frames"] = depth_frames
                    
                return result
                
            except Exception as e:
                print(f"Error processing sample {uuid}, view {camera_view}: {e}")
                
                color_frames = torch.zeros((self.num_frames, 3, *self.image_size), dtype=torch.uint8)
                
                error_result = {
                    "frames": color_frames,
                }
                
                if self.include_depth:
                    error_result["depth_frames"] = torch.zeros((self.num_frames, 3, *self.image_size), dtype=torch.uint8)
                    
                return error_result
    
    
    
    def _load_npz_data(self, npz_path: str, json_metadata: Dict, frame_indices: np.ndarray) -> Dict:
        """
        Load position and rotation data from NPZ file and sample to match frame indices.

        Args:
            npz_path: Path to NPZ file
            json_metadata: JSON metadata dictionary
            frame_indices: Frame indices used to sample the video

        Returns:
            Dictionary with 'position' and 'rotation' sampled to match video frames
        """
        npz_data = np.load(npz_path)
        dynamic_data = {}

        for key in ["position", "rotation"]:
            if key not in npz_data:
                continue

            tensor = npz_data[key]  
            T = tensor.shape[1]
            max_idx = max(frame_indices)

            if T > max_idx:
                sampled = tensor[:, frame_indices, ...] 
            else:
                valid_indices = frame_indices[frame_indices < T]
                partial = tensor[:, valid_indices, ...]
                pad_len = len(frame_indices) - len(valid_indices)

                if pad_len > 0:
                    pad = np.repeat(tensor[:, -1:, ...], pad_len, axis=1)
                    sampled = np.concatenate([partial, pad], axis=1)
                else:
                    sampled = partial

            dynamic_data[key] = sampled
        

        return dynamic_data
                
    def get_camera_views(self, uuid: str) -> List[str]:
        """
        Get available camera views for a specific UUID
        
        Args:
            uuid: Sample UUID
            
        Returns:
            List of available camera views
        """
        views = []
        for view in self.camera_views:
            color_path = os.path.join(self.video_dir, f"{uuid}_{view}_color.mp4")
            if os.path.exists(color_path):
                views.append(view)
        return views
    
    def get_sample_by_uuid(self, uuid: str, camera_view: str = None) -> Optional[Dict]:
        """
        Get a sample by UUID and optionally camera view
        
        Args:
            uuid: Sample UUID
            camera_view: Specific camera view or None for first available
            
        Returns:
            Sample dictionary or None if not found
        """
        indices = [idx for idx, data in enumerate(self.data_list) 
                  if data["uuid"] == uuid and 
                  (camera_view is None or data["camera_view"] == camera_view)]
        
        if not indices:
            return None
            
        return self.__getitem__(indices[0])
    
    

    
    def encode_global_to_vector(self, data):
        """
        Encodes JSON global data into a feature vector.

        Args:
            data: A dictionary with keys 'Scene', 'Lighting', and 'Object'

        Returns:
            A list representing the encoded feature vector
        """
        scenes = ['apt_0', 'apt_1', 'apt_2', 'apt_3', 'apt_4', 'apt_5']

        objects = [
            'uvSphereSolid',
            'frl_apartment_picture_01.object_config.json', 'frl_apartment_shoe_04.object_config.json',
            'frl_apartment_lamp_02.object_config.json', 'frl_apartment_wall_cabinet_02.object_config.json',
            'frl_apartment_vase_02.object_config.json', 'frl_apartment_bowl_01.object_config.json',
            'frl_apartment_book_02.object_config.json', 'frl_apartment_cloth_01.object_config.json',
            'frl_apartment_small_appliance_01.object_config.json', 'frl_apartment_stool_02.object_config.json',
            'frl_apartment_cushion_01.object_config.json', 'frl_apartment_cup_01.object_config.json',
            'frl_apartment_table_02.object_config.json', 'frl_apartment_clock.object_config.json',
            'frl_apartment_kitchen_utensil_03.object_config.json', 'frl_apartment_cup_02.object_config.json',
            'frl_apartment_table_01.object_config.json', 'frl_apartment_small_appliance_02.object_config.json',
            'frl_apartment_book_01.object_config.json', 'frl_apartment_monitor_stand.object_config.json',
            'frl_apartment_bowl_02.object_config.json', 'frl_apartment_vase_01.object_config.json',
            'frl_apartment_cloth_02.object_config.json', 'frl_apartment_lamp_01.object_config.json',
            'frl_apartment_picture_02.object_config.json', 'frl_apartment_umbrella.object_config.json',
            'frl_apartment_chair_05.object_config.json', 'frl_apartment_wall_cabinet_01.object_config.json',
            'frl_apartment_cloth_04.object_config.json', 'frl_apartment_tv_screen.object_config.json',
            'frl_apartment_rug_02.object_config.json', 'frl_apartment_plate_02.object_config.json',
            'frl_apartment_beanbag.object_config.json', 'frl_apartment_shoe_01.object_config.json',
            'frl_apartment_picture_04.object_config.json', 'frl_apartment_kitchen_utensil_08.object_config.json',
            'frl_apartment_kitchen_utensil_05.object_config.json', 'frl_apartment_box.object_config.json',
            'frl_apartment_tv_object.object_config.json', 'frl_apartment_shoebox_01.object_config.json',
            'frl_apartment_table_04.object_config.json', 'frl_apartment_kitchen_utensil_06.object_config.json',
            'frl_apartment_bin_03.object_config.json', 'frl_apartment_plate_01.object_config.json',
            'frl_apartment_shoe_02.object_config.json', 'frl_apartment_towel.object_config.json',
            'frl_apartment_sofa.object_config.json', 'frl_apartment_rug_01.object_config.json',
            'frl_apartment_bowl_07.object_config.json', 'frl_apartment_book_04.object_config.json',
            'frl_apartment_setupbox.object_config.json', 'frl_apartment_pan_01.object_config.json',
            'frl_apartment_choppingboard_02.object_config.json', 'frl_apartment_bowl_06.object_config.json',
            'frl_apartment_book_05.object_config.json', 'frl_apartment_bin_02.object_config.json',
            'frl_apartment_chair_01.object_config.json', 'frl_apartment_clothes_hanger_01.object_config.json',
            'frl_apartment_bike_01.object_config.json', 'frl_apartment_indoor_plant_01.object_config.json',
            'frl_apartment_shoe_03.object_config.json', 'frl_apartment_remote-control_01.object_config.json',
            'frl_apartment_clothes_hanger_02.object_config.json', 'frl_apartment_bin_01.object_config.json',
            'frl_apartment_knifeblock.object_config.json', 'frl_apartment_indoor_plant_02.object_config.json',
            'frl_apartment_sponge_dish.object_config.json', 'frl_apartment_bike_02.object_config.json',
            'frl_apartment_book_06.object_config.json', 'frl_apartment_cup_05.object_config.json',
            'frl_apartment_tvstand.object_config.json', 'frl_apartment_kitchen_utensil_09.object_config.json',
            'frl_apartment_cabinet.object_config.json', 'frl_apartment_kitchen_utensil_04.object_config.json',
            'frl_apartment_mat.object_config.json', 'frl_apartment_cushion_03.object_config.json',
            'frl_apartment_basket.object_config.json', 'frl_apartment_kitchen_utensil_02.object_config.json',
            'frl_apartment_rack_01.object_config.json', 'frl_apartment_handbag.object_config.json',
            'frl_apartment_cup_03.object_config.json', 'frl_apartment_picture_03.object_config.json',
            'frl_apartment_chair_04.object_config.json', 'frl_apartment_bowl_03.object_config.json',
            'frl_apartment_cloth_03.object_config.json', 'frl_apartment_refrigerator.object_config.json',
            'frl_apartment_book_03.object_config.json', 'frl_apartment_wall_cabinet_03.object_config.json',
            'frl_apartment_camera_02.object_config.json', 'frl_apartment_kitchen_utensil_01.object_config.json',
            'frl_apartment_table_03.object_config.json', 'frl_apartment_monitor.object_config.json',
            'banana.object_config.json', 'chefcan.object_config.json', 'skillet.object_config.json',
            'largeclamp.object_config.json', 'cheezit.object_config.json'
        ]

        vector = []

        scene = data.get("Scene", {})
        scene_name = scene.get("scene", "")
        scene_index = scenes.index(scene_name) if scene_name in scenes else -1  # -1 for unknown
        vector.append(scene_index)

        vector.append(scene.get("friction_coefficient", 0))
        vector.append(scene.get("restitution_coefficient", 0))
        vector.append(1 if scene.get("requires_lighting", False) else 0)
        vector.append(scene.get("margin", 0))
        vector.append(scene.get("gravity", 0))

        if scene.get("requires_lighting", False):
            lighting = data.get("Lighting", {})
            position = lighting.get("position", [0, 0, 0, 0])
            color = lighting.get("color", [0, 0, 0])

            vector.extend(position)
            vector.extend(color)
        else:
            vector.extend([0, 0, 0, 0])
            vector.extend([0, 0, 0])

        obj = data.get("Object", {})
        asset_path = obj.get("asset", "")
        asset_name = asset_path.split('/')[-1]
        object_index = objects.index(asset_name) if asset_name in objects else len(objects)  # last index for unknown
        vector.append(object_index)

        vector.append(obj.get("mass", 0))
        vector.append(obj.get("num_targets", 0))

        return vector


    def encode_global_to_vector_simple(self, data):
        """
        Encodes JSON global data into a feature vector.

        Args:
            data: A dictionary with keys 'Scene', 'Lighting', and 'Object'

        Returns:
            A list representing the encoded feature vector
        """
        scenes = ['apt_0', 'apt_1', 'apt_2', 'apt_3', 'apt_4', 'apt_5']

        objects = [
            'uvSphereSolid',
            'frl_apartment_picture_01.object_config.json', 'frl_apartment_shoe_04.object_config.json',
            'frl_apartment_lamp_02.object_config.json', 'frl_apartment_wall_cabinet_02.object_config.json',
            'frl_apartment_vase_02.object_config.json', 'frl_apartment_bowl_01.object_config.json',
            'frl_apartment_book_02.object_config.json', 'frl_apartment_cloth_01.object_config.json',
            'frl_apartment_small_appliance_01.object_config.json', 'frl_apartment_stool_02.object_config.json',
            'frl_apartment_cushion_01.object_config.json', 'frl_apartment_cup_01.object_config.json',
            'frl_apartment_table_02.object_config.json', 'frl_apartment_clock.object_config.json',
            'frl_apartment_kitchen_utensil_03.object_config.json', 'frl_apartment_cup_02.object_config.json',
            'frl_apartment_table_01.object_config.json', 'frl_apartment_small_appliance_02.object_config.json',
            'frl_apartment_book_01.object_config.json', 'frl_apartment_monitor_stand.object_config.json',
            'frl_apartment_bowl_02.object_config.json', 'frl_apartment_vase_01.object_config.json',
            'frl_apartment_cloth_02.object_config.json', 'frl_apartment_lamp_01.object_config.json',
            'frl_apartment_picture_02.object_config.json', 'frl_apartment_umbrella.object_config.json',
            'frl_apartment_chair_05.object_config.json', 'frl_apartment_wall_cabinet_01.object_config.json',
            'frl_apartment_cloth_04.object_config.json', 'frl_apartment_tv_screen.object_config.json',
            'frl_apartment_rug_02.object_config.json', 'frl_apartment_plate_02.object_config.json',
            'frl_apartment_beanbag.object_config.json', 'frl_apartment_shoe_01.object_config.json',
            'frl_apartment_picture_04.object_config.json', 'frl_apartment_kitchen_utensil_08.object_config.json',
            'frl_apartment_kitchen_utensil_05.object_config.json', 'frl_apartment_box.object_config.json',
            'frl_apartment_tv_object.object_config.json', 'frl_apartment_shoebox_01.object_config.json',
            'frl_apartment_table_04.object_config.json', 'frl_apartment_kitchen_utensil_06.object_config.json',
            'frl_apartment_bin_03.object_config.json', 'frl_apartment_plate_01.object_config.json',
            'frl_apartment_shoe_02.object_config.json', 'frl_apartment_towel.object_config.json',
            'frl_apartment_sofa.object_config.json', 'frl_apartment_rug_01.object_config.json',
            'frl_apartment_bowl_07.object_config.json', 'frl_apartment_book_04.object_config.json',
            'frl_apartment_setupbox.object_config.json', 'frl_apartment_pan_01.object_config.json',
            'frl_apartment_choppingboard_02.object_config.json', 'frl_apartment_bowl_06.object_config.json',
            'frl_apartment_book_05.object_config.json', 'frl_apartment_bin_02.object_config.json',
            'frl_apartment_chair_01.object_config.json', 'frl_apartment_clothes_hanger_01.object_config.json',
            'frl_apartment_bike_01.object_config.json', 'frl_apartment_indoor_plant_01.object_config.json',
            'frl_apartment_shoe_03.object_config.json', 'frl_apartment_remote-control_01.object_config.json',
            'frl_apartment_clothes_hanger_02.object_config.json', 'frl_apartment_bin_01.object_config.json',
            'frl_apartment_knifeblock.object_config.json', 'frl_apartment_indoor_plant_02.object_config.json',
            'frl_apartment_sponge_dish.object_config.json', 'frl_apartment_bike_02.object_config.json',
            'frl_apartment_book_06.object_config.json', 'frl_apartment_cup_05.object_config.json',
            'frl_apartment_tvstand.object_config.json', 'frl_apartment_kitchen_utensil_09.object_config.json',
            'frl_apartment_cabinet.object_config.json', 'frl_apartment_kitchen_utensil_04.object_config.json',
            'frl_apartment_mat.object_config.json', 'frl_apartment_cushion_03.object_config.json',
            'frl_apartment_basket.object_config.json', 'frl_apartment_kitchen_utensil_02.object_config.json',
            'frl_apartment_rack_01.object_config.json', 'frl_apartment_handbag.object_config.json',
            'frl_apartment_cup_03.object_config.json', 'frl_apartment_picture_03.object_config.json',
            'frl_apartment_chair_04.object_config.json', 'frl_apartment_bowl_03.object_config.json',
            'frl_apartment_cloth_03.object_config.json', 'frl_apartment_refrigerator.object_config.json',
            'frl_apartment_book_03.object_config.json', 'frl_apartment_wall_cabinet_03.object_config.json',
            'frl_apartment_camera_02.object_config.json', 'frl_apartment_kitchen_utensil_01.object_config.json',
            'frl_apartment_table_03.object_config.json', 'frl_apartment_monitor.object_config.json',
            'banana.object_config.json', 'chefcan.object_config.json', 'skillet.object_config.json',
            'largeclamp.object_config.json', 'cheezit.object_config.json'
        ]

        vector = []

        scene = data.get("Scene", {})
        scene_name = scene.get("scene", "")
        scene_index = scenes.index(scene_name) if scene_name in scenes else -1  # -1 for unknown
        vector.append(scene_index)

        vector.append(scene.get("gravity", 0))

        obj = data.get("Object", {})
        asset_path = obj.get("asset", "")
        asset_name = asset_path.split('/')[-1]
        object_index = objects.index(asset_name) if asset_name in objects else len(objects)  # last index for unknown
        vector.append(object_index)

        # vector.append(obj.get("mass", 0))

        return vector
    
    def merge_content_and_style_timewise(self, global_vec: np.ndarray, 
                                     position: np.ndarray, 
                                     rotation: np.ndarray) -> np.ndarray:
        """
        Merge global content vector with style info (position + rotation),
        where style is concatenated per time frame.

        Args:
            global_vec: np.ndarray of shape [D_g]
            position: np.ndarray of shape [T, D_p]
            rotation: np.ndarray of shape [T, D_r]

        Returns:
            A single flat vector of shape [D_g + T * (D_p + D_r),]
        """
        # Sanity check
        assert position.shape[0] == rotation.shape[0], "Position and rotation must have same T"

        # Concatenate per time step
        style_per_frame = np.concatenate([position, rotation], axis=1)  # shape: [T, D_p + D_r]

        # Flatten over time
        style_vector = style_per_frame.flatten()  # shape: [T * (D_p + D_r)]

        # Concatenate with global vector
        final_vector = np.concatenate([global_vec, style_vector], axis=0)  # shape: [D_g + T * (D_p + D_r)]

        return final_vector
    
    
    def merge_content_and_style_as_sequence(self, global_vec: np.ndarray,
                                        position: np.ndarray,
                                        rotation: np.ndarray) -> np.ndarray:
        """
        Merge global content vector with time-step style info as a [T, N] matrix.

        Args:
            global_vec: np.ndarray of shape [D_g]
            position: np.ndarray of shape [T, D_p]
            rotation: np.ndarray of shape [T, D_r]

        Returns:
            A matrix of shape [T, D_g + D_p + D_r]
        """
        T = position.shape[0]
        assert rotation.shape[0] == T, "Position and rotation must have same time length"

        style_per_timestep = np.concatenate([position, rotation], axis=1)
        if global_vec is None:
            return style_per_timestep
        else:
            global_vec = np.array(global_vec)
            global_repeated = np.repeat(global_vec[np.newaxis, :], T, axis=0)
            result_sequence = np.concatenate([global_repeated, style_per_timestep], axis=1)

            return result_sequence
    
    
    def merge_content_and_style_as_sequence2(
        self,
        global_vec: np.ndarray,              # [D_g]
        position: Optional[np.ndarray],      # [T, D_p] or None
        rotation: Optional[np.ndarray],      # [T, D_r] or None
        T: Optional[int] = None              # Required when both pos and rot are None.
    ) -> np.ndarray:
        """
        Merge global content vector with time-step style info as a [T, N] matrix.

        Args:
            global_vec: np.ndarray of shape [D_g]
            position: np.ndarray of shape [T, D_p] or None
            rotation: np.ndarray of shape [T, D_r] or None
            T: int, optional. If both position and rotation are None, T must be provided.

        Returns:
            np.ndarray of shape [T, D_g + D_p + D_r]
        """
        if T is None:
            if position is not None:
                T = position.shape[0]
            elif rotation is not None:
                T = rotation.shape[0]
            else:
                raise ValueError("If both `position` and `rotation` are None, `T` must be provided.")

        style_list = []
        if position is not None:
            assert position.shape[0] == T, "position.shape[0] != T"
            style_list.append(position)
        if rotation is not None:
            assert rotation.shape[0] == T, "rotation.shape[0] != T"
            style_list.append(rotation)

        style_per_timestep = np.concatenate(style_list, axis=1) if style_list else None

        if global_vec is not None:
            global_vec = np.asarray(global_vec)
            global_repeated = np.repeat(global_vec[np.newaxis, :], T, axis=0)
        else:
            global_repeated = None

        if global_repeated is not None and style_per_timestep is not None:
            return np.concatenate([global_repeated, style_per_timestep], axis=1)
        elif global_repeated is not None:
            return global_repeated
        elif style_per_timestep is not None:
            return style_per_timestep
        else:
            raise ValueError("At least one of global_vec, position, or rotation must be provided.")
