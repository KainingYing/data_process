import os 
from pathlib import Path

import mmcv
from base_dataset import BaseDataset


class lesion_grading(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/shaowenqi/taskonomy_data/medical_understanding/lesion_grading",
        "anno_path": "/mnt/petrelfs/share_data/shaowenqi/taskonomy_data/medical_understanding/lesion_grading/metadata_info.json",
        "sampling_num": 200,
        "visual_input_component": ["medical_image", ],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        metadata_info = mmcv.load(self.anno_path)

        for image_info in metadata_info["images"]:
            image_info["original_image_path"] = os.path.join(self.image_path, image_info["image_path"])
            image_info["source"] = "lesion_grading"

            self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):

        question = image_info["question"]
        if "option_D" not in image_info.keys():
            all_chocies = [image_info["option_A"], image_info["option_B"], image_info["option_C"]]
            all_chocies.remove(image_info["gt_answer"])
            num_choices = 3
        else:
            all_chocies = [image_info["option_A"], image_info["option_B"], image_info["option_C"], image_info["option_D"],]
            all_chocies.remove(image_info["gt_answer"])
            num_choices = 4

        qa_json = {
            "num_wrong_choices": num_choices - 1,
            "gt": image_info["gt_answer"],
            "question": question,
            "wrong_choices_list": all_chocies
        }
        qa_json = BaseDataset.post_process(qa_json, question=question)

        return qa_json