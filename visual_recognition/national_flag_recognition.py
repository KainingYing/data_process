from pathlib import Path
import json

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
        self.category_space = []
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
            self.category_space.append(country_name)

            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_space
    
    @staticmethod
    def generate_prompt(image_info, dataset_info):
        question = "Which country does the flag in the picture belong to?"
        from prompt.base_system_prompt import single_choice_classification_sys_prompt

        sys_prompt = single_choice_classification_sys_prompt

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_choices": 4,
                "gt_category": "America",
                "question": "Which country does the flag in the picture belong to?",
                "choice_a": "China",
                "choice_b": "America",
                "choice_c": "Australia",
                "choice_d": "England",
                "gt_choice": "b"
            },
            "query_dict": {
                "num_choices": 4,
                "gt_category": image_info["category"],
                "question": "Which country does the flag in the picture belong to?",
            }
        }

        user_prompt = json.dumps(input_json)
        return sys_prompt, user_prompt
