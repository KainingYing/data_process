import copy
from pathlib import Path

from base_dataset import BaseDataset


class weather_image_recognition(BaseDataset):
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of the weather is shown in the picture?"
        sys_prompt = '\n'.join(mmcv.list_from_file('/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/prompt/single_classification.txt'))

        input_json = {
            "question": question,
            "category_space": dataset_info["category_space"],
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "fogsmog",
                "question": question,
                "wrong_choices_list": ["glaze", "rain", "sandstorm"]
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


class MWD(BaseDataset):
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv

        num_choices = 4
        question = "What category of the weather is shown in the picture?"
    
        i = 0
        while i <= 10:
            try:
                # qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                gt = image_info["category"]
                category_space = copy.deepcopy(dataset_info["category_space"])
                category_space.remove(gt)

                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": image_info["category"],
                    "question": question,
                    "wrong_choices_list": category_space
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)
                break
            except:
                i += 1

        return qa_json
            