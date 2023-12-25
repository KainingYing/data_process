from pathlib import Path

from base_dataset import BaseDataset


class image_season_recognition(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/recognition_season/train",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = [
            "spring", "summer", "fall", "winter"
        ]
        self.dataset_info["category_space"] = self.category_space

    
    def parse_images_info(self):
        self.images_info = list()

        for season_path in Path(self.image_path).iterdir():
            season_category = season_path.name
            for image_path in season_path.iterdir():
                original_image_path = str(image_path)

                image_info = self.image_dict
                image_info.update(
                    {
                        "category": season_category,
                        "original_image_path": original_image_path
                    }
                )
                
                self.images_info.append(image_info)
                