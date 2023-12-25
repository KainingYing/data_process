from pathlib import Path

from base_dataset import BaseDataset


class sculpture_internet(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/sculpture/sculptures_internet",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            category = image_path.name
            self.category_space.append(category)

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "category": category
                }
            )

            self.images_info.append(image_info)

        self.dataset_info["category_space"] = self.category_space
        