import os
import json
import sys
import csv
import numpy as np
from PIL import Image
from pathlib import Path

import mmcv
from tqdm import tqdm

from base_dataset import BaseDataset


# class emotic(BaseDataset):
#     DATA_METAINFO = {
#         "anno_path": "/mnt/petrelfs/share_data/linyuqi/TaskOnomy_Evaluation/emotional_quotient_test/body_emotion_recognition/metadata_info.json",
#         "sampling_num": 100,
#         "dataset_description": "xxx",
#         "visual_input_component": ["natural_image", ]
#     }
    
#     def parse_dataset_info(self):
#         super().parse_dataset_info()

#     def parse_images_info(self):
#         self.images_info = []
        
#         anno_info = mmcv.load(self.anno_path)
#         self.category_space = []

#         for image_info in tqdm(anno_info["images"]):
#             if image_info["source"] != "emotic":
#                 continue

#             if not os.path.exists(image_info["original_image_path"]):
#                 continue
#             self.category_space.append(image_info["category"])
#             image_info["source"] = "emotic"
#             self.images_info.append(image_info)
        
#         self.dataset_info["category_space"] = list(set(self.category_space))
    
#     @staticmethod
#     def generate_qa(image_info, dataset_info, image_path):
#         import json
#         import mmcv

#         num_choices = 4
#         question = "What emotion is expressed in the artwork in the picture?"
#         sys_prompt = '\n'.join(mmcv.list_from_file('/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/prompt/single_classification.txt'))

#         input_json = {
#             "question": question,
#             "category_space": dataset_info["category_space"],
#             "example_dict": {
#                 "num_wrong_choices": num_choices - 1,
#                 "gt": "amusement",
#                 "question": question,
#                 "wrong_choices_list": ["disgust", "anger", "awe"]
#             },
#             "query_dict": {
#                 "num_wrong_choices": num_choices - 1,
#                 "gt": image_info["emotion"],
#                 "question": question,
#             }
#         }

#         user_prompt = json.dumps(input_json)

#         i = 0
#         while i <= 10:
#             try:
#                 qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
#                 qa_json = BaseDataset.post_process(qa_json, question=question)
#                 break
#             except:
#                 i += 1

#         return qa_json


class CAERS(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/mnt/petrelfs/share_data/linyuqi/TaskOnomy_Evaluation/emotional_quotient_test/body_emotion_recognition/metadata_info.json",
        "sampling_num": 200,
        "dataset_description": "xxx",
        "visual_input_component": ["natural_image", ]
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = []
        
        anno_info = mmcv.load(self.anno_path)
        self.category_space = []

        for image_info in tqdm(anno_info["images"]):
            if image_info["source"] != "CAERS":
                continue
            image_info["original_image_path"] = image_info["original_image_path"][1:]
            if not os.path.exists(image_info["original_image_path"]):
                continue
            self.category_space.append(image_info["category"])
            image_info["source"] = "CAERS"
            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_space))
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What emotion is expressed in the artwork in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "amusement",
                "question": question,
                "wrong_choices_list": ["disgust", "anger", "awe"]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["category"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        i = 0
        while i <= 10:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json