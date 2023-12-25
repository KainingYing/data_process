from pathlib import Path

from base_dataset import BaseDataset


class scitsr(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/TSR/SciTSR/images",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ocr/data/TSR/SciTSR/GT",
        "sampling_num": 200,
        "visual_input_component": ["text-rich_image",],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            anno_path = (Path(self.anno_path) / image_path.name).with_suffix(".txt")
        
            text = mmcv.list_from_file(anno_path)[0]

            image_info = self.image_dict(image_path)
            image_info.update(
                {
                    "text": text,
                    "original_image_path": str(image_path)
                }
            )
            self.images_info.append(image_info)
