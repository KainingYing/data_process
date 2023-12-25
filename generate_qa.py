import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import mmcv
from openai import OpenAI
from tqdm import tqdm
os.environ["OPENAI_BASE_URL"] = "https://api.openai-sb.com/v1"

from low_level_vision.depth_estimation import *
from low_level_vision.height_estimation import *

from visual_recognition.color_recognition import *
from visual_recognition.shape_recognition import *
from visual_recognition.texture_material_recognition import *
from visual_recognition.national_flag_recognition import *
from visual_recognition.fashion_recognition import *
from visual_recognition.abstract_visual_recognition import *
from visual_recognition.season_recognition import *
from visual_recognition.scene_recognition import *
from visual_recognition.film_and_television_recognition import *
from visual_recognition.painting_recognition import *
from visual_recognition.logo_and_brand_recognition import *
from visual_recognition.sculpture_recognition import *
from visual_recognition.landmark_recognition import *

from localization.small_object_detection import *
from localization.rotated_object_detection import *

from pixel_level_perception.image_matting import *
from pixel_level_perception.polygon_localization import *

from ocr.handwritten_text_recognition import *
from ocr.handwritten_mathematical_expression_recognition import *

from doc_understanding.visual_document_information_extraction import *
from doc_understanding.table_structure_recognition import *

from visual_prompt_understanding.visual_mark_understanding import *
from visual_prompt_understanding.som_recognition import *

from image2image_translate.jigsaw_puzzle_solving import *

from relation_reasoning.social_relation_recognition import *
from relation_reasoning.human_object_interaction_recognition import *
from relation_reasoning.human_interaction_understanding import *

from visual_illusion.color_assimilation import *
from visual_illusion.color_constancy import *
from visual_illusion.color_contrast import *
from visual_illusion.geometrical_perspective import *
from visual_illusion.geometrical_relativity import *

from visual_coding.eqn2latex import *
from visual_coding.screenshot2code import *
from visual_coding.sketch2code import *

from counting.counting_by_category import *
from counting.counting_by_reasoning import *
from counting.counting_by_visual_prompting import *


class QAGenerator:
    def __init__(self, dataset_config, task_name):
        self.openai_client = OpenAI()

        self.output_path = dataset_config["output_path"]
        self.metadata_info = mmcv.load(os.path.join(self.output_path, "metadata_info.json"))

        self.dataset_info_dict = {dataset_info["dataset_name"]: dataset_info for dataset_info in self.metadata_info["dataset_list"]}
        self.image_info_list = self.metadata_info["images"]

        self.generate_qa()
    
    def generate_qa(self):
        # metadata_info = mmcv.load(os.path.join(self.output_path, "metadata_info.json"))
        self.qa_info = list()

        for i, image_info in enumerate(tqdm(self.image_info_list)):
            if (Path(self.output_path) / "qa_info" / f"qa_{i}.json").exists():
                continue
            qa_json = eval(image_info["source"]).generate_qa(image_info, self.dataset_info_dict[image_info["source"]])
            # qa_json = self.openai_generate(sys_prompt, user_prompt)
            qa_json.update(image_info)
            qa_json["id"] = f"qa_{i}"
            mmcv.dump(qa_json, Path(self.output_path) / "qa_info" / f"qa_{i}.json", indent=4)

            self.qa_info.append(qa_json)

    def openai_generate(self, sys, user):
        while True:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    response_format={ "type": "json_object" },
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user}
                    ])
                output = eval(response.choices[0].message.content)
                break
            except:
                pass
        
        return output

    def save(self):
        self.exist_or_mkdir(self.output_path)
        json.dump(self.task_data_info, open(os.path.join(self.output_path, "qa_info.json"), "w"), indent=4)

    def exist_or_mkdir(self, folder_path):
        if not os.path.exists(folder_path):
            # If the folder does not exist, create it
            os.makedirs(folder_path)


def main(args):
    os.environ["OPENAI_API_KEY"] = args.openai_key
    task_name = args.task_name
    dataset_config_file = args.dataset_config

    dataset_config_file = mmcv.Config.fromfile(dataset_config_file)
    dataset_config = dataset_config_file.dataset[task_name]

    task_data = QAGenerator(dataset_config, task_name)
    # task_data.save()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Example script to parse arguments.")
    parser.add_argument('--task_name', type=str, default="small_object_detection", help='The name of the target dataet')
    parser.add_argument('--openai_key', type=str, default="xxxx", help='openai key')
    parser.add_argument('--dataset_config', type=str, default="/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/dataset_config.py", help='The path of dataset config.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
