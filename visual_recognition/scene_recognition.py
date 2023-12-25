from pathlib import Path

from base_dataset import BaseDataset


class indoor_scene_recognition(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/scene_recognition/Images",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        # self.category_space = [
        #     "spring", "summer", "fall", "winter"
        # ]
        # self.dataset_info["category_space"] = self.category_space

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        for scene_path in Path(self.image_path).iterdir():
            scene_category = scene_path.name
            self.category_space.append(scene_category)
            for image_path in scene_path.iterdir():
                original_image_path = str(image_path)

                image_info = self.image_dict
                image_info.update(
                    {
                        "category": scene_category,
                        "original_image_path": original_image_path
                    }
                )
                
                self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_space


class places365(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/scene_recognition/OpenDataLab___Places365/raw/places/data_256",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []
        for a_path in Path(self.image_path).iterdir():
            for scene_category_path in a_path.iterdir():
                scene_category = scene_category_path.name
                self.category_space.append(scene_category)

                for image_path in scene_category_path.iterdir():
                    image_info = self.image_dict
                    image_info.update(
                        {
                            "category": scene_category,
                            "original_image_path": str(image_path)
                        }
                    )
                
                    self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = self.category_space
                