from pathlib import Path

from base_dataset import BaseDataset


class sombench_flickr30k_grounding(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/SoM-Bench/flickr30k_grounding/som_images",
        "sampling_num": 200,
        "visual_input_component": ["natural_image", "visual_mark"],
        "dataset_description": "xxx"
    }
    
    def parse_images_info(self):
        import mmcv
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            if image_path.suffix == ".json":
                continue
            if image_path.stem.endswith("box"):
                anno_path = image_path.with_suffix(".json").with_stem(image_path.stem[:-5])
            else:
                anno_path = image_path.with_suffix(".json")

            anno_info = mmcv.load(anno_path)

            image_info = self.image_dict(image_path)
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "question": anno_info['caption'],
                    "mark_ids": anno_info["gt_ids"]
                }
            )
            self.images_info.append(image_info)


class sombench_refcocog_refseg(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/SoM-Bench/refcocog_refseg/som_images",
        "sampling_num": 200,
        "visual_input_component": ["natural_image", "visual_mark"],
        "dataset_description": "xxx"
    }
    
    def parse_images_info(self):
        import mmcv
        self.images_info = list()

        for image_path in Path(self.image_path).iterdir():
            if image_path.suffix == ".json":
                continue
            if image_path.stem.endswith("box"):
                anno_path = image_path.with_suffix(".json").with_stem(image_path.stem[:-5])
            else:
                anno_path = image_path.with_suffix(".json")

            anno_info = mmcv.load(anno_path)

            for anno_ in anno_info:
                image_info = self.image_dict(image_path)
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "question": anno_['text'],
                        "mark_ids": anno_["ref_id"]
                    }
                )
                self.images_info.append(image_info)