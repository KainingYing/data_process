from pathlib import Path

import numpy as np

from base_dataset import BaseDataset


class internet_poster(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/movie/internet/films",
        "sampling_num": 200,
        "url": "opendatalab",
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        for movie_category_path in Path(self.image_path).iterdir():
            movie_category = movie_category_path.name

            for movie_path in movie_category_path.iterdir():
                movie_name = movie_path.stem
                self.category_space.append(movie_name)

                image_info = self.image_dict
                image_info.update(
                    {
                        "original_image_path": str(movie_path),
                        "category": movie_name,
                        "movie_type": movie_category
                    }
                )

                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_space


class movie_posters_kaggle(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/movie/movie_posters/Multi_Label_dataset/Images",
        "anno_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/movie/movie_posters/Multi_Label_dataset/train.csv",
        "sampling_num": 200,
        "url": "opendatalab",
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import csv
        self.images_info = list()
        self.category_space = []

        anno_info_list = csv.reader(open(self.anno_path))
        next(anno_info_list)
        for row in anno_info_list:
            image_name, category_list = row[0], eval(row[1])

            original_image_path = str(Path(self.image_path) / f"{image_name}.jpg")

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "category": category_list,
                }
            )

            self.images_info.append(image_info)

            self.category_space.extend(category_list)
        
        self.dataset_info["category_space"] = list(set(self.category_space))
    