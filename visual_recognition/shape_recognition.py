from pathlib import Path

from base_dataset import BaseDataset


class twod_geometric_shapes_dataset(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/shape/2D geometric shapes dataset/output",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            shape_category = image_path.name.split("_")[0]
            self.category_space.append(shape_category)

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "category": shape_category
                }
            )
            
            self.images_info.append(image_info)

        self.dataset_info["category_space"] = self.category_space


class gpt_auto_generated_shape(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/shape/gpt_auto_generate_shape/images",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
        self.category_space = ["Circle", "Diamond", "Ellipse", "Heart Shape", "Hexagon/Pentagon"]
        self.dataset_info["category_space"] = self.category_space
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            shape_category = image_path.name.split("_")[0]
            if shape_category == "Pentagon":
                shape_category = "Hexagon/Pentagon"
            self.category_space.append(shape_category)

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "category": shape_category
                }
            )
            self.images_info.append(image_info)
