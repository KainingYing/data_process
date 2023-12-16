from pathlib import Path
import csv

import mmcv
from PIL import Image

from base_dataset import BaseDataset


class python_auto_generated_color(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/color_recognition/colors.csv",
        "save_image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/color_recognition/images",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        csv_reader = csv.reader(open(self.anno_path))

        for row in csv_reader:
            _, color_name, _, r, g, b = row

            image = Image.new("RGB", (200, 200), (int(r), int(g), int(b)))
            self.exist_or_mkdir(Path(self.save_image_path) )
            image.save(Path(self.save_image_path) / self.new_image_name(e='png'))
            rgb_category = (int(r), int(g), int(b))

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(Path(self.save_image_path) / self.new_image_name(e='png')),
                    "rgb_category": rgb_category,
                    "color_name":color_name
                }
            )
            self.images_info.append(image_info)
