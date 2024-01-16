from pathlib import Path

import mmcv

from base_dataset import BaseDataset


id_to_illusion_category = {
    1: "assimilation",
    2: "assimilation",
    3: "contrast",
    4: "contrast",
    5: "contrast",
    6: "constancy",
    7: "assimilation",
    8: "assimilation",
    9: "perspective",
    10: "relativity",
    11: "relativity",
    12: "relativity",
    13: "perspective",
    14: "assimilation",
}


class gvil_assimilation(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/visual_lllusion/GVIL-main/dataset/images",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/visual_lllusion/VL-Illusion",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image", "natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        # for 
        pair_anno_info = mmcv.load(Path(self.anno_path) / "pair_info.json")
        vqa_anno_info = mmcv.load(Path(self.anno_path) / "vqa_annotation.json")
        qa_type_list = ['samediff_qa', 'subj_qa', 'desc_qa']
        for qa_type in qa_type_list:
            pair_list = pair_anno_info[qa_type]

            for pair_images in pair_list:
                eval_id1, eval_id2 = pair_images
                category = id_to_illusion_category[int(eval_id1.split("_")[0])]

                if category != 'assimilation':
                    continue

                img1_gt = vqa_anno_info[eval_id1]["answer_match"]
                img2_gt = vqa_anno_info[eval_id2]["answer_match"]

                original_image_path_1 = str(Path(self.image_path) / vqa_anno_info[eval_id1]["img"])
                original_image_path_2 = str(Path(self.image_path) / vqa_anno_info[eval_id2]["img"])

                image_info = self.image_dict(original_image_path_1)
                image_info.update(
                    {
                        "original_image_path": [original_image_path_1, original_image_path_2],
                        "question": [vqa_anno_info[eval_id1]['question'], vqa_anno_info[eval_id2]['question']],
                        "gt": [img1_gt, img2_gt]
                    }
                )
                
                self.images_info.append(image_info)
    
    @staticmethod
    def generate_qa(image_info, dataset_info, image_path):
        import json
        import mmcv
        from openai import OpenAI

        num_choices = 2
        width = image_info["width"]
        height = image_info["height"]

        _question = image_info["question"][1]

        question = f"I will provide you an image related to visual illusions. Your task is to simulate human sensory experience to answer the following question.\nQuestion: {_question}"

        while True:
            try:
                chatgpt_prompt = "Your assignment is to construct two options for a given prompt. You will receive a question which typically has a binary choice for an answer, derived directly from the context of the question. I will provide you with one of the options. Your task is to complete the other option and return the two choices in the form of a JSON Dict format. You must contain the key of 'output_choice', which is a List."
                query_json = {
                    "example": {
                        "question": "Is the ball on the left green or cyan?",
                        "gt_option": "green",
                        "output_choice": ["green", "cyan"]
                    },
                    "query": {
                        "question": _question,
                        "gt_option": image_info["gt"][1],
                    }
                }
                options = BaseDataset.openai_generate(chatgpt_prompt, json.dumps(query_json))

                if isinstance(options, dict):
                    options = options["output_choice"]
                else:
                    pass

                assert image_info["gt"][1] in options
                options.remove(image_info["gt"][1])

                qa_json = {
                    "num_wrong_choices": num_choices - 1,
                    "gt": image_info["gt"][1],
                    "question": question,
                    "wrong_choices_list": options
                }
                qa_json = BaseDataset.post_process(qa_json, question=question)

                return qa_json

                pass
            except:
                pass

        # # prompt = 
        # # question = f"I will provide you with an image along with a question related to that image. Additionally, there will be two possible answers to choose from. Your task is to evaluate and determine which answer is better, or if it's a tie, or if both answers are inadequate.\nQuestion: {image_info['question']}\nAnswer 1: {image_info['model_outputs'][0]}\nAnswer 1: {image_info['model_outputs'][1]}"

        
        # if gt == "left":
        #     wrong_choices_list = ["right"]
        # elif gt == "right":
        #     wrong_choices_list = ["left"]
        # elif gt == "purple":
        #     wrong_choices_list = ["magenta"]
        # elif gt == "lighter":
        #     wrong_choices_list = ["darker"]
        # elif gt == "orange":
        #     if "yellow" in _question:
        #         wrong_choices_list = ["yellow"]
        #     elif "red" in _question:
        #         wrong_choices_list = ["red"]
        #     else:
        #         raise NotImplementedError
        # elif gt == "cyan":
        #     wrong_choices_list = ["green"]
        # elif gt == "magenta":
        #     wrong_choices_list = ["purple"]
        # elif gt == "darker":
        #     wrong_choices_list = ["lighter"]
        # elif gt == "red":
        #     if "orrange" in _question:
        #         wrong_choices_list = ["orrange"]
        #     else:
        #         raise NotImplementedError
        # elif gt == "green":
        #     wrong_choices_list = ["cyan"]
        # elif gt == "no":
        #     wrong_choices_list = ["yes"]
        # elif gt == "yes":
        #     wrong_choices_list = ["no"]
        # elif gt == "yellow":
        #     wrong_choices_list = ["orrange"]
        # else:
        #     raise NotImplementedError
        

        # qa_json = {
        #     "num_wrong_choices": num_choices - 1,
        #     "gt": gt,
        #     "question": question,
        #     "wrong_choices_list": wrong_choices_list
        # }
        qa_json = BaseDataset.post_process(qa_json, question=question)

        return qa_json
