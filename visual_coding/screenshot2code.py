from pathlib import Path

import mmcv

from base_dataset import BaseDataset


class pix2code_andriod(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/visual_code/screenshot2html/pix2code-master/datasets/android/all_data",
        "sampling_num": 200,
        "visual_input_component": ["screenshot_image", ],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            if image_path.suffix == '.png':
                anno_gui_path = image_path.with_suffix('.gui')

                gui_text_list = mmcv.list_from_file(anno_gui_path)
                gui_text = '\n'.join(gui_text_list)

                image_info = self.image_dict
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "text": gui_text
                    }
                )
                self.images_info.append(image_info)


class pix2code_ios(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/visual_code/screenshot2html/pix2code-master/datasets/ios/all_data",
        "sampling_num": 200,
        "visual_input_component": ["screenshot_image", ],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            if image_path.suffix == '.png':
                anno_gui_path = image_path.with_suffix('.gui')

                gui_text_list = mmcv.list_from_file(anno_gui_path)
                gui_text = '\n'.join(gui_text_list)

                image_info = self.image_dict
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "text": gui_text
                    }
                )
                self.images_info.append(image_info)


class pix2code_web(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/visual_code/screenshot2html/pix2code-master/datasets/web/all_data",
        "sampling_num": 200,
        "visual_input_component": ["screenshot_image", ],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            if image_path.suffix == '.png':
                anno_gui_path = image_path.with_suffix('.gui')

                gui_text_list = mmcv.list_from_file(anno_gui_path)
                gui_text = '\n'.join(gui_text_list)

                image_info = self.image_dict
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "text": gui_text
                    }
                )
                self.images_info.append(image_info)
