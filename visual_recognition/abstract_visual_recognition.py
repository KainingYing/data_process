from pathlib import Path

import struct
import numpy as np

from base_dataset import BaseDataset


class quickdraw(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/OpenDataLab___QuickDraw-Extended/raw/QuickDraw_sketches_final",
        "sampling_num": 200,
        "url": "opendatalab",
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        # self.category_space = [
        #     "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        # ]
        # self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        for image_category in Path(self.image_path).iterdir():
            category_name = image_category.name
            self.category_space.append(category_name)

            for image_path in image_category.iterdir():
                original_image_path = str(image_path)

                image_info = self.image_dict
                image_info.update(
                    {
                        "original_image_path": original_image_path,
                        "category": category_name
                    }
                )

                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_space
        pass
    