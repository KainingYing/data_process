from pathlib import Path
import csv

import mmcv
from PIL import Image
import struct
import numpy as np

from base_dataset import BaseDataset


def read_mnist_images(image_file):
    with open(image_file, 'rb') as file:
        # 读取文件头的4个整数，使用大端格式
        magic, num_images, rows, cols = struct.unpack('>IIII', file.read(16))
        
        # 读取图像数据，每个图像有rows * cols个字节
        images = np.fromfile(file, dtype=np.uint8).reshape(num_images, rows, cols)
        
    return images

def read_mnist_labels(label_file):
    with open(label_file, 'rb') as file:
        # 读取文件头的两个整数：魔数和标签的数量
        magic, num_labels = struct.unpack(">II", file.read(8))
        
        # 读取标签数据，每个标签占一个字节
        labels = np.fromfile(file, dtype=np.uint8)
    
    return labels


class fashion_mnist(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/fashion/OpenDataLab___Fashion-MNIST/raw/t10k-images-idx3-ubyte",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/fashion/OpenDataLab___Fashion-MNIST/raw/t10k-labels-idx1-ubyte",
        "save_image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/fashion_recognition/images",
        "sampling_num": 50,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        self.images_info = list()

        image_numpy = read_mnist_images(self.image_path)
        label_numpy = read_mnist_labels(self.anno_path)

        for i in range(image_numpy.shape[0]):
            image = image_numpy[i]
            label = label_numpy[i]

            original_image_path = Path(self.save_image_path) / self.new_image_name(e='png')
            self.exist_or_mkdir(Path(self.save_image_path))
            self.save_image(image, original_image_path)

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "category": self.category_space[label]
                }
            )
            self.images_info.append(image_info)


class deepfashion(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/fashion/deepfashion/img",
        "sampling_num": 150,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        # self.category_space = [
        #     "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        # ]
        # self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for fashion_category in Path(self.image_path).iterdir():
            category = fashion_category.name.replace("_", " ")
            self.category_space.append(category)
            
            for image_path in fashion_category.iterdir():
                original_image_path = str(image_path)
                image_info = self.image_dict

                image_info.update(
                    {
                        "original_image_path": original_image_path,
                        "category": category
                    }
                )
            
        self.dataset_info["category_space"] = self.category_space
