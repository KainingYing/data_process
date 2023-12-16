import sys
import numpy as np
from PIL import Image
from pathlib import Path
import copy

import mmcv
from shapely.geometry import Polygon

from base_dataset import BaseDataset


class hicodet(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/hico_20160224_det/annotations/test_hico.json",
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/hico_20160224_det/images/test2015",
        "image_save_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/relation_reasoning/human-object_interaction_recognition",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = [
            'adjust', 'assemble', 'block', 'blow', 'board', 'break',
            'brush_with', 'buy', 'carry', 'catch', 'chase', 'check',
            'clean', 'control', 'cook', 'cut', 'cut_with', 'direct',
            'drag', 'dribble', 'drink_with', 'drive', 'dry', 'eat', 'eat_at',
            'exit', 'feed', 'fill', 'flip', 'flush', 'fly', 'greet', 'grind',
            'groom', 'herd', 'hit', 'hold', 'hop_on', 'hose',
            'hug', 'hunt', 'inspect', 'install', 'jump', 'kick', 'kiss',
            'lasso', 'launch', 'lick', 'lie_on', 'lift', 'light',
            'load', 'lose', 'make', 'milk', 'move', 'no_interaction',
            'open', 'operate', 'pack', 'paint', 'park', 'pay',
            'peel', 'pet', 'pick', 'pick_up', 'point', 'pour', 'pull', 'push',
            'race', 'read', 'release', 'repair', 'ride',
            'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear', 'sign',
            'sip', 'sit_at', 'sit_on', 'slide', 'smell', 'spin', 'squeeze',
            'stab', 'stand_on', 'stand_under', 'stick', 'stir', 'stop_at', 'straddle', 'swing', 'tag',
            'talk_on', 'teach', 'text_on', 'throw', 'tie', 'toast', 'train', 'turn', 'type_on', 'walk', 'wash',
            'watch', 'wave', 'wear', 'wield', 'zip'
        ]
        self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        self.images_info = list()
        json_data_list = mmcv.load(self.anno_path)

        # t = 0
        for image_info in json_data_list:
            # image_info_ = dict()
            ori_image_path = Path(self.image_path) / image_info["file_name"]

            if len(image_info["hoi_annotation"]) != 1:
                continue

            subject_id = image_info["hoi_annotation"][0]["subject_id"]
            object_id = image_info["hoi_annotation"][0]["object_id"]

            subject_bbox = image_info["annotations"][subject_id]['bbox']
            # subject_category = 'person'
            if image_info["annotations"][object_id]['category_id'] == 1:
                # 过滤掉人与人的交互
                continue
            object_bbox = image_info["annotations"][object_id]['bbox']

            category_id = image_info["hoi_annotation"][0]["category_id"]
            category = self.category_space[category_id - 1]

            # _image_info = copy.deepcopy(self.image_dict)
            image_dict = self.image_dict
            image_dict.update({
                "ori_image_path": ori_image_path,
                "category": category,
                "bounding_box_coordinates": {
                "subject": subject_bbox,
                "object": object_bbox}
            })
            self.images_info.append(image_dict)


class vcoco(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/v-coco/annotations/test_vcoco.json",
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/coco/val2014",
        "image_save_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/relation_reasoning/human-object_interaction_recognition",
        "sampling_num": 100,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
        }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = [
            'carry_obj', 'catch_obj', 'cut_instr',
            'cut_obj',
            'drink_instr',
            'eat_instr',
            'eat_obj',
            'hit_instr',
            'hit_obj',
            'hold_obj',
            'jump_instr',
            'kick_obj',
            'lay_instr',
            'look_obj',
            'point_instr',
            'read_obj',
            'ride_instr',
            'run',
            'sit_instr',
            'skateboard_instr',
            'ski_instr',
            'smile',
            'snowboard_instr',
            'stand',
            'surf_instr',
            'talk_on_phone_instr',
            'throw_obj',
            'walk',
            'work_on_computer_instr']
        
        self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        self.images_info = []
        json_data_list = mmcv.load(self.anno_path)

        for image_info in json_data_list:
            image_info_ = dict()

            ori_image_path = Path(self.image_path) / image_info["file_name"]

            if len(image_info["hoi_annotation"]) != 1 or image_info["hoi_annotation"][0]["object_id"] == -1:
                continue

            subject_id = image_info["hoi_annotation"][0]["subject_id"]
            object_id = image_info["hoi_annotation"][0]["object_id"]

            subject_bbox = image_info["annotations"][subject_id]['bbox']
            # subject_category = 'person'
            object_bbox = image_info["annotations"][object_id]['bbox']

            category_id = image_info["hoi_annotation"][0]["category_id"]
            category = self.category_space[category_id - 1]

            image_info_["ori_image_path"] = [ori_image_path]
            image_info_["caetegory"] = category
            image_info_["source"] = "hicodet"
            image_info_["visual_input_component"] = ["natural_image"]
            image_info_["bounding_box_coordinates"] = {
                "subject": subject_bbox,
                "object": object_bbox
            }

            # self.data_info.append(image_info_)

    def sample(self):
        pass


if __name__ == '__main__':
    # hicodet_data = hicodet()
    vcoco_data = vcoco()
