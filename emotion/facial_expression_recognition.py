import os
import json
import sys
import csv
import numpy as np
from PIL import Image
sys.path.append("data_process")
from pathlib import Path

import mmcv

from base_dataset import MergeDataset, BaseDataset


class ferplus(BaseDataset):
    METADATA_INFO = {
        "anno_path": "data/face_emotion_recognition/ferplus/train.csv",
        "image_save_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/emotion/fer"
    }
    def __init__(self):
        self.anno_path = self.METADATA_INFO["anno_path"]
        self.image_save_path = self.METADATA_INFO["image_save_path"]

        self.parse_dataset()
        self.data_info = self.get_data_info()
    
    def parse_dataset(self):
        self.dataset_info = dict()
        self.dataset_info["category_space"] = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    def get_data_info(self):
        self.data_info = list()
        csv_reader = csv.reader(open(self.METADATA_INFO["anno_path"]))
        next(csv_reader, None)
        for row in csv_reader:
            image_info = dict()

            category, imgbytes = row

            category = self.dataset_info["category_space"][int(category)]
            # 将像素值字符串拆分为数字列表，并将其转换为numpy数组
            pixel_values = np.array(imgbytes.split(), dtype=np.uint8)
            
            # 假设图像是48x48像素的灰度图像，可以根据需求调整尺寸和通道数
            image = pixel_values.reshape((48, 48))
            
            # 将numpy数组转换为Pillow图像对象
            image = Image.fromarray(image)

            image_info["category"] = category

            save_image_file, save_image_path = self.get_image_name("png")
            image_info["image_file"] = save_image_file

            self.exist_or_mkdir(str(Path(save_image_path).parent))
            image.save(save_image_path)
            image_info["visual_input_component"] = ["natural_image", "grayscale", "low-resolution"]
            image_info["source"] = ferplus
        
    def sample(self):
        pass


class ferg_db(BaseDataset):
    METADATA_INFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/face_emotion_recognition/FERG_DB_256",
        "image_save_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/emotion/fer"
    }
    def __init__(self):
        self.image_path = self.METADATA_INFO["image_path"]
        self.image_save_path = self.METADATA_INFO["image_save_path"]

        self.parse_dataset()
        self.data_info = self.get_data_info()
    
    def parse_dataset(self):
        self.dataset_info = dict()
        self.dataset_info["category_space"] = ["Angry", "Disgust", "Fear", "Joy", "Netural", "Sadness", "Surprise"]

    def get_data_info(self):
        self.data_info = list()
        for actor_path in Path(self.image_path).iterdir():
            for actor_emotion_path in actor_path.iterdir():
                for acrtor_emotion_image in actor_emotion_path.iterdir():
                    ori_image_path = str(acrtor_emotion_image)
                    actor, category = actor_emotion_path.name.split("_")
                    category = category.capitalize()

                    image_info = {}
                    image_info["ori_image_path"] = ori_image_path
                    image_info["actor"] = actor
                    image_info["category"] = category
                    image_info["source"] = "ferg_db"
                    image_info["visual_input_component"] = ["synthetic_image"]

                    self.data_info.append(image_info)
        
    def sample(self):
        pass


if __name__ == '__main__':
    ferplus_data = ferplus()
    ferg_db_data = ferg_db()
