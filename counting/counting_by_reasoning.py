from pathlib import Path
import csv
import json

import mmcv
import numpy as np
from PIL import Image
from iso3166 import countries

from base_dataset import BaseDataset


class tallyqa_complex(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/visual_genome",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/counting/tallyqa/test.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info_lsit = mmcv.load(self.anno_path)
        # cate_anno_list = mmcv.list_from_file(self.category_map)
        for anno_info in anno_info_lsit:
            if anno_info["issimple"]:
                continue
            original_image_path = Path(self.image_path) / anno_info['image']
            answer = anno_info["answer"]
            question = anno_info["question"]

            image_info = self.image_dict
            image_info.update(
                {
                    "question": question,
                    "answer": answer,
                    "original_image_path": str(original_image_path)
                }
            )
            self.images_info.append(image_info)
