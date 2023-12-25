from pathlib import Path


from base_dataset import BaseDataset


class am2k(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/image_matting/am-2k/validation/original",
        "trimap_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/image_matting/am-2k/validation/trimap",
        "mask_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/image_matting/am-2k/validation/mask",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path  in Path(self.image_path).iterdir():
            image_name = image_path.name

            original_image_path = str(image_path)

            trimap_path = str((Path(self.trimap_path) / image_name).with_suffix('.png'))
            mask_path = str((Path(self.mask_path) / image_name).with_suffix('.png'))

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "trimap_path": trimap_path,
                    "mask_path": mask_path
                }
            )

            self.images_info.append(image_info)

    
class aim500(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/image_matting/aim-500/AIM-500/original",
        "trimap_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/image_matting/aim-500/AIM-500/trimap",
        "mask_path": "/mnt/petrelfs/yingkaining/yingkaining/lvlm_evaluation/data/image_matting/aim-500/AIM-500/mask",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()

        for image_path  in Path(self.image_path).iterdir():
            image_name = image_path.name

            original_image_path = str(image_path)

            trimap_path = str((Path(self.trimap_path) / image_name).with_suffix('.png'))
            mask_path = str((Path(self.mask_path) / image_name).with_suffix('.png'))

            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": original_image_path,
                    "trimap_path": trimap_path,
                    "mask_path": mask_path
                }
            )
            
            self.images_info.append(image_info)
        