from pathlib import Path

import mmcv

from base_dataset import BaseDataset


class taskonomy(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/depth_data/taskonomy",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info_list = mmcv.load(Path(self.image_path) / "annotations.json")

        for anno_info in anno_info_list:
            rgb_path = Path(self.image_path) / "rgb" / anno_info["rgb"]
            depth_path = Path(self.image_path) / "depth" / anno_info["depth"]

            image_info = self.image_dict
            image_info.update(
                {
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "cam_in": anno_info["cam_in"],
                    "depth_metric": anno_info["depth_metric"],
                    "max": anno_info["max"],
                }
            )

            self.images_info.append(image_info)


class nyu(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/depth_data/nyu",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info_list = mmcv.load(Path(self.image_path) / "annotations.json")

        for anno_info in anno_info_list:
            rgb_path = Path(self.image_path) / "rgb" / anno_info["rgb"]
            depth_path = Path(self.image_path) / "depth" / anno_info["depth"]

            image_info = self.image_dict
            image_info.update(
                {
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "cam_in": anno_info["cam_in"],
                    "depth_metric": 1000.,
                }
            )

            self.images_info.append(image_info)


class nuscenes(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/depth_data/nuscenes",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info_list = mmcv.load(Path(self.image_path) / "annotations.json")

        for anno_info in anno_info_list:
            rgb_path = Path(self.image_path) / "rgb" / anno_info["rgb"]
            depth_path = Path(self.image_path) / "depth" / anno_info["depth"]

            image_info = self.image_dict
            image_info.update(
                {
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "cam_in": anno_info["cam_in"],
                    "depth_metric": 256.,
                }
            )

            self.images_info.append(image_info)


class kitti(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/depth_data/kitti",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_info_list = mmcv.load(Path(self.image_path) / "annotations.json")

        for anno_info in anno_info_list:
            rgb_path = Path(self.image_path) / "rgb" / anno_info["rgb"]
            depth_path = Path(self.image_path) / "depth" / anno_info["depth"]

            image_info = self.image_dict
            image_info.update(
                {
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "cam_in": anno_info["cam_in"],
                    "depth_metric": 256.,
                }
            )

            self.images_info.append(image_info)
