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

            try:
                image_info = self.image_dict(str(original_image_path))
            except:
                pass
            image_info.update(
                {
                    "original_image_path": str(original_image_path),
                    "text": eq_text
                }
            )

            self.images_info.append(image_info)

    @staticmethod
    def generate_qa(image_info, dataset_info, save_image_path):
        import json
        import mmcv

        num_choices = 4
        question = "Convert the equation in this image to LaTeX code."
        sys_prompt = '\n'.join(mmcv.list_from_file('/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/prompt/handwritten_mathematical_expression_recognition.txt'))

        input_json = {
            "question": question,
            "example_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": "\\angle A F E = 3 0 ^ { \\circ } , A E = 1 2",
                "question": question,
                "wrong_choices_list": [
                        "\\angle AFE = 30^{\\circ}, AE = 21",
                        "\\angle AFE = 60^{\\circ}, AE = 12",
                        "\\angle AFE = 30^{\\circ}, AF = 12"
                    ]
            },
            "query_dict": {
                "num_wrong_choices": num_choices - 1,
                "gt": image_info["text"],
                "question": question,
            }
        }

        user_prompt = json.dumps(input_json)

        while True:
            try:
                qa_json = BaseDataset.openai_generate(sys_prompt, user_prompt)
                qa_json = BaseDataset.post_process(qa_json)
                break
            except:
                pass

        return qa_json