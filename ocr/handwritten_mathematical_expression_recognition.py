from pathlib import Path

import struct
import numpy as np
import mmcv

from base_dataset import BaseDataset


class hme100k(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/HMER/HME100K/images",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/HMER/HME100K/GT.txt",
        "sampling_num": 100,
        "visual_input_component": ["text-rich_image", "natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_list = mmcv.list_from_file(self.anno_path)
        for image_anno in anno_list:
            image_name, text = image_anno.split('\t')

            original_image_path = Path(self.image_path) / image_name

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "text": text
                }
            )

            self.images_info.append(image_info)


class crohme2014(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/HMER/CROHME2014/images",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/HMER/CROHME2014/GT.txt",
        "sampling_num": 100,
        "visual_input_component": ["synthetic_image", "text-rich_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_list = mmcv.list_from_file(self.anno_path)
        for image_anno in anno_list:
            image_name, text = image_anno.split('\t')

            original_image_path = Path(self.image_path) / image_name

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "text": text
                }
            )

            self.images_info.append(image_info)
    