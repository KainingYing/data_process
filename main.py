import os
import json
import random
import argparse
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

import mmcv
from prompt.utils import *

from task_zoo import *


def check_function(func):
    try:
        eval(func)
        function_defined = True
    except (AttributeError, NameError):
        function_defined = False
    
    return function_defined


class MergeDataset:
    def __init__(self, dataset_config, task_name):
        self.output_path = dataset_config["output_path"]

        self.task_name = task_name
        self.dataset_list = []
        self.json_mode = False
        if dataset_config.get("dataset_list", None) is None and dataset_config.get("metadata_info", None) is not None:
            self.json_mode = True
            self.samping_num = dataset_config["sampling_num"]
            self.parse_metadata_info(dataset_config["metadata_info"])
        else:
            for dataset_name in dataset_config["dataset_list"]:
                _dataset = eval(dataset_name)()
                self.dataset_list.append(_dataset)
    
    def parse_metadata_info(self, metadata_info_path):
        raw_metadata_info = mmcv.load(metadata_info_path)

        dataset_image_info_list = defaultdict(list)
        for image_info in raw_metadata_info["images"]:
            if check_function(f"{image_info['source']}.process_raw_metadata_info"):
                if eval(image_info["source"]).process_raw_metadata_info(image_info):
                    dataset_image_info_list[image_info["source"]].append(eval(image_info["source"]).process_raw_metadata_info(image_info))
                else:
                    continue
            else:
                dataset_image_info_list[image_info["source"]].append(image_info)
            
        images_info = []
        for key, value in dataset_image_info_list.items():
            sampling_num = self.samping_num[key]
            if sampling_num > len(value):
                sampling_num = len(value)
            
            output_list = random.sample(value, sampling_num)
            print(f"{key.ljust(10)} {sampling_num}")
            images_info.extend(output_list)
        
        for dataset_info in raw_metadata_info["dataset_list"]:
            dataset_info["sampling_num"] = self.samping_num[dataset_info["dataset_name"]]
        self.task_data_info = raw_metadata_info
        self.task_data_info["images"] = images_info

        for image_info in self.task_data_info["images"]:
            original_image_path = image_info["original_image_path"]
            if isinstance(original_image_path, list):
                original_image_path = original_image_path[0]
            elif isinstance(original_image_path, str):
                pass
            else:
                raise NotImplementedError
            with Image.open(original_image_path) as img:
                width, height = img.size
            image_info["width"] = width
            image_info["height"] = height
        
        print((f"{'overall'.ljust(10)} {len(self.task_data_info['images'])}"))
    
    def merge(self):
        if not self.json_mode:
            self.task_data_info = dict()
            self.task_data_info["task_name"] = self.task_name
            self.task_data_info["dataset_list"] = []
            self.task_data_info["images"] = []

            for dataset in self.dataset_list:
                dataset_info = dataset.dataset_info
                
                self.task_data_info["dataset_list"].append(dataset_info)
                self.task_data_info["images"].extend(dataset.sample())

                print(f"{dataset_info['dataset_name'].ljust(10)} {len(dataset.sample())}")
            
            print((f"{'overall'.ljust(10)} {len(self.task_data_info['images'])}"))

            for image_info in self.task_data_info["images"]:

                original_image_path = image_info["original_image_path"]
                if isinstance(original_image_path, list):
                    original_image_path = original_image_path[0]
                elif isinstance(original_image_path, str):
                    pass
                else:
                    raise NotImplementedError
                with Image.open(original_image_path) as img:
                    width, height = img.size
                image_info["width"] = width
                image_info["height"] = height
        else:
            pass

    def save(self):
        self.exist_or_mkdir(self.output_path)
        json.dump(self.task_data_info, open(os.path.join(self.output_path, "metadata_info.json"), "w"), indent=4)

    def exist_or_mkdir(self, folder_path):
        if not os.path.exists(folder_path):
            # If the folder does not exist, create it
            os.makedirs(folder_path)


def main(args):
    task_name = args.task_name
    dataset_config_file = args.dataset_config

    dataset_config_file = mmcv.Config.fromfile(dataset_config_file)
    dataset_config = dataset_config_file.dataset[task_name]

    task_data = MergeDataset(dataset_config, task_name)
    task_data.merge()
    task_data.save()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Example script to parse arguments.")
    parser.add_argument('--task_name', type=str, default="person_reid", help='The name of the target dataet')
    parser.add_argument('--dataset_config', type=str, default="/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/dataset_config.py", help='The path of dataset config.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
