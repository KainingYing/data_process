from pathlib import Path

import mmcv

from base_dataset import BaseDataset


class social_relation_dataset(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/OpenDataLab___Social_Relation/raw/interpersonal_relation_dataset/testing.txt",
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/OpenDataLab___Social_Relation/raw/interpersonal_relation_dataset/img",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        a = mmcv.load("/mnt/petrelfs/yingkaining/data/markllava/finetune/visual_genome/train.json")
        super().parse_dataset_info()
        self.category_space = [
            'dominated', 'assured', 'demonstrative', 'involved', 'friendly', 'warm', 'trusting', 'competitive'
        ]
        self.dataset_info["category_space"] = self.category_space
        self.dataset_info["category_num"] = "multiple"
    
    def parse_images_info(self):
        self.images_info = list()
        anno_data_list = mmcv.list_from_file(self.anno_path)
        # json_data_list = mmcv.load(self.anno_path)
        for anno_data in anno_data_list:
            anno_list = anno_data.strip().split()
            original_image_path = Path(self.image_path) / anno_list[0]
            width, height = self.get_image_width_height(original_image_path)

            x, y, w, h = [float(i) for i in anno_list[1:5]]
            face_1_box = [x / width, y / height, (x + w) / width, (y + h) / height]

            x, y, w, h = [int(i) for i in anno_list[5:9]]
            face_2_box = [x / width, y / height, (x + w) / width, (y + h) / height]

            label_list = anno_list[9: ]
            label_list = [int(i) for i in label_list]

            category_list = [self.category_space[i] for i, flag in enumerate(label_list) if flag == 1]

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "bounding_box_coordinates": [face_1_box, face_2_box],
                    "category_list": category_list
                }
            )

