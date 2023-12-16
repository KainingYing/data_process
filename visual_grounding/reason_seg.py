import os
import json
import sys
import csv
import numpy as np
from PIL import Image
sys.path.append("data_process")
from pathlib import Path

import mmcv
from shapely.geometry import Polygon

from base_dataset import BaseDataset


class reason_seg(BaseDataset):
    METADATA_INFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ReasonSeg/val",
        "image_save_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_grounding/reason_seg"
    }
    def __init__(self):
        self.image_path = self.METADATA_INFO["image_path"]
        self.image_save_path = self.METADATA_INFO["image_save_path"]

        self.parse_dataset()
        self.get_data_info()
    
    def parse_dataset(self):
        self.dataset_info = dict()

    def get_data_info(self):
        self.data_info = list()
        for image_path in Path(self.image_path).iterdir():
            image_info = dict()
            if image_path.suffix == ".json":
                continue
            try:
                anno_json = image_path.with_suffix(".json")
                anno_info = mmcv.load(anno_json)
                question = anno_info["text"][0]
                polygon_coords = anno_info["shapes"][0]["points"]
            except:
                continue

            polygon = Polygon(polygon_coords)
            # 获取多边形的最小包围框的左上角和右下角坐标
            x1, y1, x2, y2 = polygon.bounds

            bounding_box_coordinates = [x1, y1, x2, y2]

            img = Image.open(image_path)
            # 获取图片的大小（宽度和高度）
            width, height = img.size
            bounding_box_coordinates = [x1 / width, y1 / height, x2 / width, y2 / height]

            image_info["bounding_box_coordinates"] = [bounding_box_coordinates]
            image_info["question"] = question
            image_info["image_file"] = [str(image_path)]

            self.data_info.append(image_info)

        pass

    def sample(self):
        pass


if __name__ == '__main__':
    reason_seg_data = reason_seg()
