from pathlib import Path

import mmcv

from base_dataset import BaseDataset


class dota(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/rotate/dota/images/images",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/rotate/dota/labelTxt-v1.5",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            anno_path = Path(self.anno_path) / f"{image_path.stem}.txt"
            anno_info_list = mmcv.list_from_file(anno_path)

            bbox_list = []
            category_list = []
            for instance_anno_info in anno_info_list[2:]:
                x1, y1, x2, y2, x3, y3, x4, y4, category, difficult = instance_anno_info.split()
                bbox_list.append([x1, y1, x2, y2, x3, y3, x4, y4])
                category_list.append(category)

                self.category_space.append(category)

            width, height = self.get_image_width_height(image_path)
            
            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "bounding_box_coordinates": bbox_list,
                    "category": category_list
                }
            )

            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_space))


class ssdd_inshore(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/rotate/ssdd/test/inshore/images",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/rotate/ssdd/test/inshore/labelTxt",
        "sampling_num": 200,
        "visual_input_component": ["sar_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            anno_path = Path(self.anno_path) / f"{image_path.stem}.txt"
            anno_info_list = mmcv.list_from_file(anno_path)

            bbox_list = []
            category_list = []
            for instance_anno_info in anno_info_list:
                x1, y1, x2, y2, x3, y3, x4, y4, category, difficult = instance_anno_info.split()
                bbox_list.append([x1, y1, x2, y2, x3, y3, x4, y4])
                category_list.append(category)

                self.category_space.append(category)

            width, height = self.get_image_width_height(image_path)
            
            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "bounding_box_coordinates": bbox_list,
                    "category": category_list
                }
            )

            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_space))


class ssdd_offshore(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/rotate/ssdd/test/offshore/images",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/rotate/ssdd/test/offshore/labelTxt",
        "sampling_num": 200,
        "visual_input_component": ["sar_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()

    def parse_images_info(self):
        self.images_info = list()
        self.category_space = []

        for image_path in Path(self.image_path).iterdir():
            anno_path = Path(self.anno_path) / f"{image_path.stem}.txt"
            anno_info_list = mmcv.list_from_file(anno_path)

            bbox_list = []
            category_list = []
            for instance_anno_info in anno_info_list:
                x1, y1, x2, y2, x3, y3, x4, y4, category, difficult = instance_anno_info.split()
                bbox_list.append([x1, y1, x2, y2, x3, y3, x4, y4])
                category_list.append(category)

                self.category_space.append(category)

            width, height = self.get_image_width_height(image_path)
            
            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "bounding_box_coordinates": bbox_list,
                    "category": category_list
                }
            )

            self.images_info.append(image_info)
        
        self.dataset_info["category_space"] = list(set(self.category_space))