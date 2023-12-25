from pathlib import Path

import mmcv
import numpy as np

from base_dataset import BaseDataset


class fsc147(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/counting/images_384_VarV2",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/counting/annotation_FSC147_384.json",
        "density_map": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/counting/gt_density_map_adaptive_384_VarV2",
        "sampling_num": 200,
        "visual_input_component": ["natural_image", "visual_mark"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info_lsit = mmcv.load(self.anno_path)
        for image_name, image_info_dict in anno_info_lsit.items():
            box_examples = image_info_dict["box_examples_coordinates"]

            counting_num = int(np.load((Path(self.density_map) / image_name).with_suffix('.npy')).sum())

            image_info = self.image_dict
            image_info.update(
                {
                    "box_examples": box_examples,
                    "counting_num": counting_num,
                    "original_image_path": str(Path(self.image_path) / image_name)
                }
            )
            self.images_info.append(image_info)
