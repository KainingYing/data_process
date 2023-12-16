from pathlib import Path
import json

import mmcv

from base_dataset import BaseDataset


class vipbench(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/keypoint_detection/coco/test2017",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ViP-Bench/vip-bench-meta-data.json",
        "image_output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/image-to-image_translation/jigsaw_puzzle_solving/mscoco",
        "metainfo_output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/image-to-image_translation/jigsaw_puzzle_solving",
        "sampling_num": 200
    }
    
    def parse_images_info(self):
        self.images_info = list()

        json_data_info = mmcv.load(self.anno_path)
        bbox_path = Path(self.anno_path).parent / "bbox"
        human_path = Path(self.anno_path).parent / "human"

        bbox_question_dict = dict()
        with open(bbox_path / "questions.jsonl", 'r') as file:
            for line in file:
                # 将每一行解析为 JSON 对象
                data = json.loads(line)
                bbox_question_dict[data["image"]] = data

        human_question_dict = dict()
        with open(human_path / "questions.jsonl", 'r') as file:
            for line in file:
                # 将每一行解析为 JSON 对象
                data = json.loads(line)
                human_question_dict[data["image"]] = data

        for _, image_info in json_data_info.items():
            pass
            image_source = image_info["image_source"]
            image = image_info["image"]
            question = image_info["question"]
            answer = image_info["answer"]

            # bbox
            pass
            bbox_ori_image_path = bbox_path / "images" / image
            bbox_ori_question = bbox_question_dict[image]['text']

            self.images_info.append({
                "ori_image_path": str(bbox_ori_image_path),
                "question": bbox_ori_question,
                "answer": answer,
                "source": "vipbench",
                "visual_input_component": ["natural_image", "visual_mark"],
                "filter_key":{
                    "type": "bbox",
                    "image_source": image_source
                }
            })
        
            # human
            human_ori_image_path = human_path / "images" / image
            human_ori_question = human_question_dict[image]['text']

            self.images_info.append({
                "ori_image_path": str(human_ori_image_path),
                "question": human_ori_question,
                "answer": answer,
                "source": "vipbench",
                "visual_input_component": ["natural_image", "visual_mark"],
                "filter_key":{
                    "type": "bbox",
                    "image_source": image_source
                }
            })

    def sample(self):
        import random
        return random.sample(self.images_info, self.sampling_num)
