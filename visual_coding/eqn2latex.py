from pathlib import Path

import mmcv

from base_dataset import BaseDataset


class im2latex90k(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/visual_code/formula_images",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/visual_code/step0/",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image", "text-rich_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        anno_list = mmcv.list_from_file(Path(self.anno_path) / "im2latex_rendered_map.txt")
        gt_anno_list = mmcv.list_from_file(Path(self.anno_path) / "im2latex_formulas_rendered.txt")

        for anno_line in anno_list:
            image_id, image_name, _ = anno_line.split(' ')

            original_image_path = Path(self.image_path) / f"{image_name}.png"
            
            eq_text = gt_anno_list[int(image_id)]

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "text": eq_text
                }
            )

            self.images_info.append(image_info)
