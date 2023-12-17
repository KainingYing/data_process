from pathlib import Path

import mmcv

from base_dataset import BaseDataset


id_to_illusion_category = {
    1: "assimilation",
    2: "assimilation",
    3: "contrast",
    4: "contrast",
    5: "contrast",
    6: "constancy",
    7: "assimilation",
    8: "assimilation",
    9: "perspective",
    10: "relativity",
    11: "relativity",
    12: "relativity",
    13: "perspective",
    14: "assimilation",
}


class gvil_relativity(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/visual_lllusion/GVIL-main/dataset/images",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/visual_lllusion/VL-Illusion",
        "sampling_num": 200,
        "visual_input_component": ["synthetic_image", "natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        # for 
        pair_anno_info = mmcv.load(Path(self.anno_path) / "pair_info.json")
        vqa_anno_info = mmcv.load(Path(self.anno_path) / "vqa_annotation.json")
        qa_type_list = ['samediff_qa', 'subj_qa', 'desc_qa']
        for qa_type in qa_type_list:
            pair_list = pair_anno_info[qa_type]

            for pair_images in pair_list:
                eval_id1, eval_id2 = pair_images
                category = id_to_illusion_category[int(eval_id1.split("_")[0])]

                if category != 'relativity':
                    continue

                img1_gt = vqa_anno_info[eval_id1]["answer_match"]
                img2_gt = vqa_anno_info[eval_id2]["answer_mismatch"]

                original_image_path_1 = str(Path(self.image_path) / vqa_anno_info[eval_id1]["img"])
                original_image_path_2 = str(Path(self.image_path) / vqa_anno_info[eval_id2]["img"])

                image_info = self.image_dict
                image_info.update(
                    {
                        "original_image_path": [original_image_path_1, original_image_path_2],
                        "gt": [img1_gt, img2_gt]
                    }
                )
                
                self.images_info.append(image_info)
        pass 