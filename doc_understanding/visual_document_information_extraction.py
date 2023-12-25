from pathlib import Path
import csv
import json

import mmcv
import numpy as np
from PIL import Image
from iso3166 import countries

from base_dataset import BaseDataset


class funsd(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/VIE/FUNSD/images",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/VIE/FUNSD/en.val.kv.json",
        "sampling_num": 200,
        "visual_input_component": ["text-rich_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info_lsit = mmcv.load(self.anno_path)
        # cate_anno_list = mmcv.list_from_file(self.category_map)
        for image_name, anno_info in anno_info_lsit.items():
            pass
            image_path = Path(self.image_path) / image_name

            image_info = self.image_dict(image_path)
            image_info.update(
                {
                    "key_information": anno_info,
                    "original_image_path": str(image_path)
                }
            )
            self.images_info.append(image_info)
