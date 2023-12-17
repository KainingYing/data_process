from pathlib import Path

import struct
import numpy as np
import mmcv

from base_dataset import BaseDataset


class iam_line(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/HTR/IAM/images_line",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/HTR/IAM/GT_line",
        "sampling_num": 100,
        "visual_input_component": ["text-rich_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            original_image_path = str(image_path)
            
            text_path = Path(self.anno_path) / f"{image_path.stem}.txt"
            pass

            text_list = mmcv.list_from_file(text_path)
            assert len(text_list) == 1
            text = text_list[0]

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "text": text
                }
            )
            self.images_info.append(image_info)


class iam_page(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/HTR/IAM/images_page",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/HTR/IAM/GT_page",
        "sampling_num": 100,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            original_image_path = str(image_path)
            
            text_path = Path(self.anno_path) / f"{image_path.stem}.txt"

            text_list = mmcv.list_from_file(text_path)

            text = "\n".join(text_list)
            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "text": text
                }
            )
            self.images_info.append(image_info)
    