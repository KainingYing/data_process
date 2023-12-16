from pathlib import Path
import csv

import mmcv
from PIL import Image
from iso3166 import countries

from base_dataset import BaseDataset


class country_flag(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/country_flag",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        country_code_list = []
        for country in Path(self.image_path).iterdir():
            country_code = country.stem
            country_code_list.append(country_code)
            try:
                country_name = countries.get(country_code.upper()).name
            except:
                continue

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(country),
                    "category": country_name
                }
            )
