from pathlib import Path
import csv
import json

import mmcv
import numpy as np
from PIL import Image
from iso3166 import countries

from base_dataset import BaseDataset


class fsc147_category(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/counting/images_384_VarV2",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/counting/annotation_FSC147_384.json",
        "density_map": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/counting/gt_density_map_adaptive_384_VarV2",
        "category_map": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/counting/ImageClasses_FSC147.txt",
        "sampling_num": 200,
        "visual_input_component": ["natural_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        # anno_info_lsit = mmcv.load(self.anno_path)
        cate_anno_list = mmcv.list_from_file(self.category_map)
        for anno_line in cate_anno_list:
            image_name, category = anno_line.split('\t')

            counting_num = int(np.load((Path(self.density_map) / image_name).with_suffix('.npy')).sum())

            image_info = self.image_dict
            image_info.update(
                {
                    "category": category,
                    "counting_num": counting_num,
                    "original_image_path": str(Path(self.image_path) / image_name)
                }
            )
            self.images_info.append(image_info)

class countqa_vqa(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/coco/val2014",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/counting/coco_counting/CountQA_VQA_data.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info_lsit = mmcv.load(self.anno_path)

        for anno_info in anno_info_lsit:
            question = anno_info["Question"]
            answer = anno_info["Answer"]
            original_image_path = Path(self.image_path) / anno_info["Image_ID"]

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "question": question,
                    "answer": answer
                }
            )

            self.images_info.append(image_info)


class countqa_cocoqa(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/coco/val2014",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/counting/coco_counting/CountQA_COCOQA_data.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info_lsit = mmcv.load(self.anno_path)

        for anno_info in anno_info_lsit:
            question = anno_info["Question"]
            answer = anno_info["Answer"]
            original_image_path = Path(self.image_path) / anno_info["Image_ID"]

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "question": question,
                    "answer": answer
                }
            )

            self.images_info.append(image_info)
    

class tallyqa_simple(BaseDataset):
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
            if not anno_info["issimple"]:
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
