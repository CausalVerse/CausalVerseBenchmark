import os
import json
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_modules.video_dataset import VideoDataset


def _normalize_fixed_robotics_latent_view(latent_view: str) -> str:
    alias = {
        "front": "front",
        "frontview": "front",
        "side": "side",
        "sideview": "side",
        "bird": "bird",
        "birdview": "bird",
        "agent": "agent",
        "agentview": "agent",
        "eye": "robot0_eye_in_hand",
        "robot0_eye_in_hand": "robot0_eye_in_hand",
    }
    return alias.get(latent_view, latent_view)


def _available_latent_views(root_dir: str) -> List[str]:
    latent_root = os.path.join(root_dir, "latent")
    if not os.path.isdir(latent_root):
        return []
    return sorted(
        name for name in os.listdir(latent_root)
        if os.path.isdir(os.path.join(latent_root, name))
    )

class FixedRoboticsDataset(VideoDataset):
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
        """
        self.camera_views = camera_views if camera_views else ["agentview", "birdview", "frontview", "robot0_eye_in_hand", "sideview"]
        self.only_latents = only_latents
        self.latent_view = _normalize_fixed_robotics_latent_view(latent_view)
        self.meta_dir = os.path.join(root_dir, "meta")
        self.video_dir = os.path.join(root_dir, "video")
        self.latents_dir = os.path.join(root_dir, "latent", self.latent_view)
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
                latent_root = os.path.join(self.root_dir, "latent")
                views = _available_latent_views(self.root_dir)
                print(f"Error: Latent directory not found: {self.latents_dir}")
                if os.path.isdir(latent_root):
                    if views:
                        print(f"Available latent views under {latent_root}: {', '.join(views)}")
                    else:
                        print(f"Latent root exists but contains no view directories: {latent_root}")
                else:
                    print(f"Latent root directory not found: {latent_root}")
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
                        try:
                            latent_data = torch.load(latent_path, map_location="cpu")
                            latents = latent_data.get("latents", None)
                            if latents is None:
                                continue
                            if latents.shape[0] != self.num_frames:
                                continue
                            frame_indices = latent_data.get("frame_numbers", list(range(self.num_frames)))
                        except Exception as e:
                            print(f"[Error] Failed to load {latent_path}: {e}")
                            continue
                        sample_data = {
                            "uuid": uuid,
                            "sample_idx": sample_idx,
                            "latent_path": latent_path,
                            "npz_path": npz_path,
                            "json_path": json_path,
                            "latents": latents,  
                            "frame_indices": frame_indices 
                        }
                        self.data_list.append(sample_data)
            else:
                for view in self.camera_views:
                    color_video_path = os.path.join(self.video_dir, f"{uuid}_{view}.mp4")
                    if not os.path.exists(color_video_path):
                        continue
                    
                    video_metadata = self._get_video_metadata(color_video_path)
                    if not video_metadata:
                        continue
                    
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
                mean = sample_data["latents"] 
                frame_indices = sample_data["frame_indices"] 

                json_metadata = self.uuid_to_meta[uuid]
                npz_data = self._load_npz_data(sample_data["npz_path"], json_metadata, np.array(frame_indices))
                
                global_vec = self.encode_global_to_vector_simple(json_metadata.get("Global", {}))
                
                vector = self.merge_content_and_style_as_sequence(
                    global_vec,
                    npz_data["joint_pos"],
                    npz_data["joint_vel"],
                    npz_data["eef_pos"],
                    npz_data["eef_quat"],
                    npz_data["gripper_qpos"],
                    npz_data["gripper_qvel"],
                    npz_data["pos_objects"],
                    npz_data["rot_objects"]
                )
                
                result = {
                    "latent": mean,                    
                    "vector": vector,               
                    "frame_indices": frame_indices  
                }
                return result
                
            except Exception as e:
                print(f"Error processing latents for sample {uuid}: {e}")
                
                # Return a placeholder sample.
                error_result = {
                    "mean": torch.zeros((16, 4, 64, 64)),
                    "vector": torch.zeros((16, 6)), 
                    "frame_indices": list(range(16))
                }
                return error_result 
        else:
            clip_idx = sample_data.get("clip_idx", 0)

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
                global_vec = self.encode_global_to_vector_compact(json_metadata.get("Global", {}))
                vector = self.merge_content_and_style_as_sequence(
                    None,
                    npz_data["joint_pos"],
                    npz_data["joint_vel"],
                    npz_data["eef_pos"],
                    npz_data["eef_quat"],
                    npz_data["gripper_qpos"],
                    npz_data["gripper_qvel"],
                )

                color_frames = color_frames.permute(1, 0, 2, 3)  
                result = {
                    "frames": color_frames,                                
                    "vector": vector,                                         
                }
                if depth_frames is not None:
                    result["depth_frames"] = depth_frames
                    
                return result
                
            except Exception as e:
                # Return placeholder on error
                print(f"Error processing sample {uuid}: {e}")
                
                # Return black frames with correct dimensions
                color_frames = torch.zeros((self.num_frames, 3, *self.image_size), dtype=torch.uint8)
                
                error_result = {
                    "frames": color_frames,
                }
                
                return error_result
        
    
    
    def _load_npz_data(self, npz_path: str, json_metadata: Dict, frame_indices: np.ndarray) -> Dict:
        """
        Load dynamic data from NPZ file based on JSON metadata and sample to match frame indices.
        For pos_objects and rot_objects, pad N to 8 if necessary.

        Args:
            npz_path: Path to NPZ file
            json_metadata: JSON metadata dictionary
            frame_indices: Frame indices used to sample the video

        Returns:
            Dictionary of dynamic data variables sampled to match the video frames
        """
        try:
            npz_data = np.load(npz_path)
            dynamic_data = {}

            expected_keys = [
                "joint_pos", "joint_vel",
                "eef_pos", "eef_quat",
                "gripper_qpos", "gripper_qvel",
                "pos_objects", "rot_objects"
            ]

            for key in expected_keys:
                if key not in npz_data:
                    continue

                data = npz_data[key]

                if data.ndim == 2: 
                    T = data.shape[0]
                    if T > max(frame_indices):
                        dynamic_data[key] = data[frame_indices]
                    else:
                        valid_indices = frame_indices[frame_indices < T]
                        partial = data[valid_indices]
                        pad_len = len(frame_indices) - len(valid_indices)
                        if pad_len > 0:
                            last_frame = data[-1:]
                            pad = np.repeat(last_frame, pad_len, axis=0)
                            dynamic_data[key] = np.concatenate([partial, pad], axis=0)
                        else:
                            dynamic_data[key] = partial

                elif data.ndim == 3:  
                    N, T, D = data.shape

                    # Step 1: sample time
                    if T > max(frame_indices):
                        sampled = data[:, frame_indices, :]
                    else:
                        valid_indices = frame_indices[frame_indices < T]
                        partial = data[:, valid_indices, :]
                        pad_len = len(frame_indices) - len(valid_indices)
                        if pad_len > 0:
                            last_frame = data[:, -1:, :].repeat(pad_len, axis=1)
                            sampled = np.concatenate([partial, last_frame], axis=1)
                        else:
                            sampled = partial  # shape [N, len(frame_indices), D]

                    # Step 2: pad objects to N=8 if needed
                    if N < 8:
                        pad_N = 8 - N
                        pad_shape = (pad_N, sampled.shape[1], sampled.shape[2])
                        pad = np.zeros(pad_shape, dtype=sampled.dtype)
                        sampled = np.concatenate([sampled, pad], axis=0)
                    elif N > 8:
                        sampled = sampled[:8, :, :]

                    dynamic_data[key] = sampled

                else:
                    print(f"Skipping key {key} with unsupported shape {data.shape}")
                    continue

            return dynamic_data

        except Exception as e:
            print(f"Error loading and sampling NPZ data from {npz_path}: {e}")
            return {}
    
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
        # Find all indices with this UUID
        indices = [idx for idx, data in enumerate(self.data_list) 
                  if data["uuid"] == uuid and 
                  (camera_view is None or data["camera_view"] == camera_view)]
        
        if not indices:
            return None
            
        # Return first match (or specific camera view if specified)
        return self.__getitem__(indices[0])
    
    

    
    def encode_global_to_vector(self, data):
        impedance_modes = ["fixed", "variable"]
        object_categories = ['akita_black_bowl', 'alphabet_soup', 'basin_faucet', 'basin_faucet_base', 'basin_faucet_movable', 
                            'basket', 'bbq_sauce', 'black_book', 'bowl_drainer', 'butter', 'chefmate_8_frypan', 
                            'chocolate_pudding', 'cookies', 'cream_cheese', 'desk_caddy', 'dining_set_group', 
                            'flat_stove', 'glazed_rim_porcelain_ramekin', 'ketchup', 'macaroni_and_cheese', 
                            'meshes', 'microwave', 'milk', 'moka_pot', 'new_salad_dressing', 'orange_juice', 
                            'plate', 'popcorn', 'porcelain_mug', 'red_bowl', 'red_coffee_mug', 'salad_dressing', 
                            'short_cabinet', 'short_fridge', 'simple_rack', 'textures', 'tomato_sauce', 
                            'white_bowl', 'white_cabinet', 'white_storage_box', 'white_yellow_mug', 
                            'wine_bottle', 'wine_rack', 'wine_rack_stand', 'wooden_cabinet', 'wooden_cabinet_base', 
                            'wooden_shelf', 'wooden_tray', 'wooden_two_layer_shelf', 'yellow_book']
        
        texture_categories = ['brown_ceramic_tile', 'canvas_sky_blue', 'capriccio_sky', 'ceramic', 'cream-plaster', 
                            'dapper_gray_floor', 'dark_blue_wall', 'dark_floor_texture', 'dark_gray_plaster', 
                            'dark_green_plaster_wall', 'gray_ceramic_tile', 'gray_floor', 'gray_plaster', 
                            'gray_wall', 'grigia_caldera_porcelain_floor', 'kona_gotham', 'light-gray-floor-tile', 
                            'light-gray-plaster', 'light_blue_wall', 'light_floor', 'light_gray_plaster', 
                            'light_grey_plaster', 'marble_floor', 'martin_novak_wood_table', 'meeka-beige-plaster', 
                            'new_light_gray_plaster', 'rustic_floor', 'seamless_wood_planks_floor', 
                            'smooth_light_gray_plaster', 'stucco_wall', 'table_light_wood', 
                            'tile_grigia_caldera_porcelain_floor', 'white_marble_floor', 'white_wall', 
                            'yellow_linen_wall_texture']

        encoded = []

        robots = data["Robots"]
        imp_mode = robots.get("impedance_mode", "fixed")
        imp_index = impedance_modes.index(imp_mode) if imp_mode in impedance_modes else 0
        encoded.append(float(imp_index) / max(len(impedance_modes) - 1, 1))
        encoded.append(float(robots["damping_ratio"]))

        materials = data["Scene"]["material"]
        for mat_key in ["floorplane", "table_texture", "table_legs", "walls_mat", "table_mat"]:
            mat = materials.get(mat_key, {})
            reflectance = mat.get("reflectance")
            shininess = mat.get("shininess")
            specular = mat.get("specular")

            reflectance = float(reflectance) if isinstance(reflectance, (int, float)) else 0.0
            shininess = float(shininess) if isinstance(shininess, (int, float)) else 0.0
            specular = float(specular) if isinstance(specular, (int, float)) else 0.0
            encoded.extend([reflectance, shininess, specular])

            texture = mat.get("texture", {})
            texture_name = ""
            if "file" in texture: 
                file_path = texture["file"]
                texture_name = file_path.split("/")[-1].split(".")[0]  
            texture_index = texture_categories.index(texture_name) if texture_name in texture_categories else -1
            texture_value = float(texture_index) / (len(texture_categories) - 1) if texture_index >= 0 else -1.0
            encoded.append(texture_value)

        lights = data["Scene"]["light"]
        for light_key in ["light0", "light1"]:
            light = lights.get(light_key, {})
            diffuse = light.get("diffuse", [0.0, 0.0, 0.0])
            direction = light.get("dir", [0.0, 0.0, 0.0])
            pos = light.get("pos", [0.0, 0.0, 0.0])
            castshadow = 1.0 if light.get("castshadow", False) else 0.0

            specular = light.get("specular")
            directional = light.get("directional")

            specular = float(specular) if isinstance(specular, (int, float)) else 0.0
            directional = float(directional) if isinstance(directional, (int, float)) else 0.0

            encoded.extend([float(x) for x in diffuse])
            encoded.extend([float(x) for x in direction])
            encoded.extend([float(x) for x in pos])
            encoded.extend([castshadow, specular, directional])

        objects = data.get("Object", [])[:2]  
        for i in range(2):  
            if i < len(objects):
                obj_name = objects[i]["path"].split("/")[-1]
                obj_index = object_categories.index(obj_name) if obj_name in object_categories else -1
                obj_value = float(obj_index) / (len(object_categories) - 1) if obj_index >= 0 else -1.0
            else:
                obj_value = -1.0  # use -1.0 to indicate missing object
            encoded.append(obj_value)

        return encoded

    def encode_global_to_vector_simple(self, data):
        object_categories = ['akita_black_bowl', 'alphabet_soup', 'basin_faucet', 'basin_faucet_base', 'basin_faucet_movable', 
                            'basket', 'bbq_sauce', 'black_book', 'bowl_drainer', 'butter', 'chefmate_8_frypan', 
                            'chocolate_pudding', 'cookies', 'cream_cheese', 'desk_caddy', 'dining_set_group', 
                            'flat_stove', 'glazed_rim_porcelain_ramekin', 'ketchup', 'macaroni_and_cheese', 
                            'meshes', 'microwave', 'milk', 'moka_pot', 'new_salad_dressing', 'orange_juice', 
                            'plate', 'popcorn', 'porcelain_mug', 'red_bowl', 'red_coffee_mug', 'salad_dressing', 
                            'short_cabinet', 'short_fridge', 'simple_rack', 'textures', 'tomato_sauce', 
                            'white_bowl', 'white_cabinet', 'white_storage_box', 'white_yellow_mug', 
                            'wine_bottle', 'wine_rack', 'wine_rack_stand', 'wooden_cabinet', 'wooden_cabinet_base', 
                            'wooden_shelf', 'wooden_tray', 'wooden_two_layer_shelf', 'yellow_book']
        
        encoded = []

        objects = data.get("Object", [])[:4]  # Keep at most the first 4 objects.
        for i in range(4):  # Fill 4 object slots.
            if i < len(objects):
                obj_name = objects[i]["path"].split("/")[-1]
                obj_index = object_categories.index(obj_name) if obj_name in object_categories else -1
                obj_value = float(obj_index) / (len(object_categories) - 1) if obj_index >= 0 else -1.0
            else:
                obj_value = -1.0  # Use -1.0 for a missing object.
            encoded.append(obj_value)

        return encoded
    
    
    def merge_content_and_style_as_sequence(self,
                                        global_vec: np.ndarray,
                                        *dynamic_vars: np.ndarray) -> np.ndarray:
        """
        Merge global content vector with time-step style info as a [T, N] matrix.

        Args:
            global_vec: np.ndarray of shape [D_g] or None
            *dynamic_vars: Variable number of np.ndarrays, each of shape [T, D_i] or [N, T, D_j]

        Returns:
            A matrix of shape [T, D_g + sum(D_i)] if global_vec is not None,
            else [T, sum(D_i)]
        """
        processed_vars = []
        target_N = 4
        for var in dynamic_vars:
            if var.ndim == 1:
                raise ValueError(f"Unexpected 1D variable: {var.shape}")
            elif var.ndim == 2:
                # [T, D]
                processed_vars.append(var)
            elif var.ndim == 3:
                # [N, T, D] → reshape to [T, N*D]
                N, T, D = var.shape
                if N > target_N:
                    var = var[:target_N]  # Truncate.
                elif N < target_N:
                    pad_shape = (target_N - N, T, D)
                    pad = np.zeros(pad_shape, dtype=var.dtype)
                    var = np.concatenate([var, pad], axis=0)  # Zero-pad.
                var_reshaped = var.transpose(1, 0, 2).reshape(T, target_N * D)
                processed_vars.append(var_reshaped)
            else:
                raise ValueError(f"Unsupported variable shape: {var.shape}")

        T = processed_vars[0].shape[0]
        for var in processed_vars:
            assert var.shape[0] == T, "All dynamic variables must share same T"

        style_per_timestep = np.concatenate(processed_vars, axis=1)

        if global_vec is None:
            # Return the style matrix directly.
            return style_per_timestep
        else:
            global_vec = np.array(global_vec)
            global_repeated = np.repeat(global_vec[np.newaxis, :], T, axis=0)
            return np.concatenate([global_repeated, style_per_timestep], axis=1)
