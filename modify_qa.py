import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import mmcv
from openai import OpenAI
from tqdm import tqdm
os.environ["OPENAI_BASE_URL"] = "https://api.openai-sb.com/v1"

from prompt.track_progress_rich import track_progress_rich

from task_zoo import *


def generate_qa_single(image_info, output_path, dataset_info, idx):
    if not (Path(output_path) / "qa_info" / f"qa_{idx}.json").exists():
        return
    qa_json = eval(image_info["source"]).modify_qa(mmcv.load(Path(output_path) / "qa_info" / f"qa_{idx}.json"))
    if qa_json is None:
        pass
    else:
        if "question" in  image_info.keys():
            del image_info["question"]
        
        for key, value in image_info.items():
            if key not in qa_json:
                qa_json[key] = value
        # qa_json.update(image_info)
        qa_json["id"] = f"qa_{idx}"
        mmcv.dump(qa_json, Path(output_path) / "qa_info" / f"qa_{idx}.json", indent=4)


class QAGenerator:
    def __init__(self, dataset_config, task_name, nproc=1):
        self.openai_client = OpenAI()

        self.output_path = dataset_config["output_path"]
        self.metadata_info = mmcv.load(os.path.join(self.output_path, "metadata_info.json"))

        self.dataset_info_dict = {dataset_info["dataset_name"]: dataset_info for dataset_info in self.metadata_info["dataset_list"]}
        self.image_info_list = self.metadata_info["images"]
        self.nproc = nproc

        self.generate_qa()
    
    def generate_qa(self):
        self.qa_info = list()
        input_list = []
        for i in range(len(self.image_info_list)):
            input_list.append((self.image_info_list[i], self.output_path, self.dataset_info_dict[self.image_info_list[i]["source"]]))
        results = track_progress_rich(generate_qa_single, input_list, nproc=self.nproc, chunksize=self.nproc)

        # for i, image_info in enumerate(tqdm(self.image_info_list)):
        #     if (Path(self.output_path) / "qa_info" / f"qa_{i}.json").exists():
        #         continue 
        #     qa_json = eval(image_info["source"]).generate_qa(image_info, self.dataset_info_dict[image_info["source"]], os.path.join(self.output_path, "images"))
        #     qa_json.update(image_info)
        #     qa_json["id"] = f"qa_{i}"
        #     mmcv.dump(qa_json, Path(self.output_path) / "qa_info" / f"qa_{i}.json", indent=4)

        #     self.qa_info.append(qa_json)
    
    def generate_qa_single(self, image_info, idx):
        if (Path(self.output_path) / "qa_info" / f"qa_{idx}.json").exists():
            return
        qa_json = eval(image_info["source"]).generate_qa(image_info, self.dataset_info_dict[image_info["source"]], os.path.join(self.output_path, "images"))
        if qa_json is None:
            pass
        else:
            qa_json.update(image_info)
            qa_json["id"] = f"qa_{idx}"
            mmcv.dump(qa_json, Path(self.output_path) / "qa_info" / f"qa_{idx}.json", indent=4)

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

    task_data = QAGenerator(dataset_config, task_name, args.nproc)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Example script to parse arguments.")
    parser.add_argument('--task_name', type=str, default="image_captioning", help='The name of the target dataet')
    parser.add_argument('--openai_key', type=str, default="sb-fb43570969a6107c4dc146c41841f9c342f9c94befc8fd29", help='openai key')
    parser.add_argument('--dataset_config', type=str, default="/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/dataset_config.py", help='The path of dataset config.')
    parser.add_argument('--nproc', type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
