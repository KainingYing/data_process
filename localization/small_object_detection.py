from pathlib import Path
from collections import defaultdict

import mmcv

from base_dataset import BaseDataset


class sod4bird(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/sod/sod4bird/train/images",
        "anno_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/sod/sod4bird/train/annotations/split_val_coco.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image", "high_resolution"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        anno_info = mmcv.load(self.anno_path)

        image2annotations = defaultdict(list)

        for annotation_info in anno_info["annotations"]:
            image2annotations[annotation_info["image_id"]].append(annotation_info)

        for image_anno_info in anno_info["images"]:
            image_id = image_anno_info["id"]
            if len(image2annotations[image_id]) >= 4:
                continue

            bounding_box_coordinates = []
            for anno in image2annotations[image_id]:
                bounding_box_coordinates.append(anno["bbox"])
            
            image_info = self.image_dict(str(Path(self.image_path) / image_anno_info["file_name"]))
            image_info.update(
                {
                    "original_image_path": str(Path(self.image_path) / image_anno_info["file_name"]),
                    "bounding_box_coordinates": bounding_box_coordinates,
                    "type": "xywh"
                }
            )
            self.images_info.append(image_info)
        
    @staticmethod
    def generate_qa(image_info, dataset_info):
        import json
        from prompt.base_system_prompt import multi_object_detection_without_category_sys_prompt

        sys_prompt = multi_object_detection_without_category_sys_prompt
        input_json = {
            "question": "Please detect all the birds in this image.",
            "example_dict": {
                "num_choices": 4,
                "image_width":"",
                "gt_bounding_box_coordinates": "",
                "question": "What brand is shown in the logo depicted in the picture?",
                "choice_a": "tesla",
                "choice_b": "byd",
                "choice_c": "bmw",
                "choice_d": "benz",
                "gt_choice": "b"
            },
            "query_dict": {
                "num_choices": 4,
                "gt_category": image_info["category"],
                "question": "What brand is shown in the logo depicted in the picture?",
            }
        }

        user_prompt = json.dumps(input_json)

        qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
        return qa_json


class drone2021(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/sod/drone2021/images",
        "anno_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/sod/drone2021/annotations/split_val_coco.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image", "high_resolution"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        anno_info = mmcv.load(self.anno_path)

        image2annotations = defaultdict(list)

        for annotation_info in anno_info["annotations"]:
            image2annotations[annotation_info["image_id"]].append(annotation_info)

        for image_anno_info in anno_info["images"]:
            image_id = image_anno_info["id"]
            if len(image2annotations[image_id]) >= 4:
                continue
            bounding_box_coordinates = []
            for anno in image2annotations[image_id]:
                bounding_box_coordinates.append(anno["bbox"])
            
            image_info = self.image_dict(str(Path(self.image_path) / image_anno_info["file_name"]))
            image_info.update(
                {
                    "original_image_path": str(Path(self.image_path) / image_anno_info["file_name"]),
                    "bounding_box_coordinates": bounding_box_coordinates,
                    "type": "xywh"
                }
            )
            self.images_info.append(image_info)


class tinyperson(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/sod/tiny_set/test",
        "anno_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/sod/tiny_set/annotations/tiny_set_test.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        anno_info = mmcv.load(self.anno_path)

        image2annotations = defaultdict(list)

        for annotation_info in anno_info["annotations"]:
            image2annotations[annotation_info["image_id"]].append(annotation_info)

        for image_anno_info in anno_info["images"]:
            image_id = image_anno_info["id"]
            if len(image2annotations[image_id]) >= 4:
                continue
            bounding_box_coordinates = []
            for anno in image2annotations[image_id]:
                bounding_box_coordinates.append(anno["bbox"])
            
            image_info = self.image_dict(str(Path(self.image_path) / image_anno_info["file_name"]))
            image_info.update(
                {
                    "original_image_path": str(Path(self.image_path) / image_anno_info["file_name"]),
                    "bounding_box_coordinates": bounding_box_coordinates,
                    "type": "xywh"
                }
            )
            self.images_info.append(image_info)   
  