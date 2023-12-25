from pathlib import Path

import numpy as np

from base_dataset import BaseDataset


class wikiart(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/wikiart/wikiart/data/train-00000-of-00072.parquet",
        "anno_file_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/wikiart/wikiart/dataset_infos.json",
        "save_image_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/painting_recognition/wikiart",
        "sampling_num": 200,
        "visual_input_component": ["painting_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import pandas as pd
        import mmcv
        self.images_info = list()

        df = pd.read_parquet(self.anno_path)
        anno_file = mmcv.load(self.anno_file_path)

        artist_list = anno_file["huggan--wikiart"]["features"]["artist"]
        genre_list =  anno_file["huggan--wikiart"]["features"]["genre"]
        style_list =  anno_file["huggan--wikiart"]["features"]["style"]

        self.images_info = list()

        for _, row in enumerate(df.itertuples()):

            image_rgb = mmcv.imfrombytes(row.image["bytes"])
            original_image_path = Path(self.save_image_path) / self.new_image_name()
            self.save_rgb_image(image_rgb, original_image_path)
            artist = artist_list['names'][row.artist]
            genre = genre_list['names'][row.genre]
            style = style_list['names'][row.style]

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "artist": artist,
                    "genre": genre,
                    "style": style
                }
            )

            self.images_info.append(image_info)
        
        self.dataset_info["artist_list"] = artist_list["names"]
        self.dataset_info["genre_list"] = genre_list["names"]
        self.dataset_info["style_list"] = style_list["names"]


class best_artwork_of_all_time(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/painting/best_artwork_of_all_time/images/images",
        "sampling_num": 200,
        "visual_input_component": ["painting_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        self.category_lsit = []
        for artist_path in Path(self.image_path).iterdir():
            artist_name = " ".join(artist_path.stem.split("_"))
            self.category_lsit.append(artist_name)
            for image_path in artist_path.iterdir():

                image_info = self.image_dict
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "catgeory": artist_name
                    }
                )
                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_lsit


class van_gogh_paintings_dataset(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/painting/van_gogh_paintings_dataset",
        "sampling_num": 200,
        "visual_input_component": ["painting_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import re
        self.images_info = list()
        self.category_space = []

        for painting_path in Path(self.image_path).iterdir():
            for image_path in painting_path.iterdir():
                if bool(re.search(r'\d$', image_path.stem)):
                    continue

                image_info = self.image_dict
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "category": image_path.stem
                    }
                )
                self.images_info.append(image_info)
                self.category_space.append(image_path.stem)
        
        self.dataset_info["category_space"] = self.category_lsit


class chinese_patinting_internet(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/painting/chinese_painting_internet/",
        "sampling_num": 200,
        "visual_input_component": ["painting_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import re
        self.images_info = list()
        self.category_space = []

        for painting_path in Path(self.image_path).iterdir():
            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(painting_path),
                    "category": painting_path.stem
                }
            )
            self.images_info.append(image_info)
            self.category_space.append(painting_path.stem)
        
        self.dataset_info["category_space"] = self.category_lsit
