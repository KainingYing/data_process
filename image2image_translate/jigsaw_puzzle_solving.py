import mmcv
import uuid
from pathlib import Path
import sys
sys.path.append("data_process")
from PIL import Image
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import pyarrow.parquet as pq


from base_dataset import BaseDataset


from PIL import Image, ImageDraw
import random


def process_image_v3(image_name):
    # Load the image
    if type(image_name) == str:
        img = Image.open(image_name)
    else:
        img = Image.fromarray(image_name)

    # Define patch size
    patch_size = (img.width // 4, img.height // 4)

    # Create a list to store patches
    patches = []
    for i in range(4):
        for j in range(4):
            # Extract patch
            patch = img.crop((j * patch_size[0], i * patch_size[1],
                              (j + 1) * patch_size[0], (i + 1) * patch_size[1]))
            patches.append(patch)

    # Randomly shuffle the middle 2x2 patches
    middle_patches = patches[5:7] + patches[9:11]
    ori_index = list(range(4))
    random.shuffle(ori_index)
    middle_patches = [middle_patches[i] for i in ori_index]

    correct_index = [ori_index.index(i) + 1 for i in range(4)]

    # random.shuffle(middle_patches)

    # Replace the shuffled patches back
    patches[5:7], patches[9:11] = middle_patches[:2], middle_patches[2:]

    # Draw borders and add numbers with increased font size
    draw = ImageDraw.Draw(img)
    # Calculate font size based on patch size (larger than before)
    font_size = min(patch_size) // 3
    font = ImageFont.truetype("/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/Arial.ttf", font_size)

    t = 1
    for i, patch in enumerate(patches):
        x, y = (i % 4) * patch_size[0], (i // 4) * patch_size[1]
        img.paste(patch, (x, y))

        # Draw border
        draw.rectangle([x, y, x + patch_size[0], y + patch_size[1]], outline="blue", width=3)

        # Add numbers to middle patches with increased font size
        if 4 < i < 11 and i not in [7, 8]:
            text = str(t)
            t += 1
            text_width, text_height = 0, 0
            # Center the text
            text_x = x + (patch_size[0] - 3) / 2
            text_y = y + (patch_size[1] - 3) / 2
            draw.text((text_x, text_y), text, fill=(0, 255, 0), font=font)

    return img, correct_index


class jigsaw_puzzle_solving_natural(BaseDataset):
    # collect patch shuffled data from natural images
    # coco
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/keypoint_detection/coco/test2017",
        "image_output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/image-to-image_translation/jigsaw_puzzle_solving/mscoco",
        "sampling_num": 100,
        "visual_input_component": ["natural_image", "visual_mark"],
        "dataset_description": "xxx"
    }

    def parse_images_info(self):
        self.images_info = list()
        
        t = 0
        for i, image_path in enumerate(Path(self.image_path).iterdir()):
            if i < 100:
                continue
            if t >= 200:
                break
            t += 1
            shuffled_image, correct_indices = process_image_v3(str(image_path))
            new_image_name = self.new_image_name("jpg")
            self.exist_or_mkdir(Path(self.image_output_path))
            shuffled_image.save(Path(self.image_output_path) / new_image_name)

            self.images_info.append({
                "ori_image_path": str(Path(self.image_output_path) / new_image_name),
                "source": self.dataset_name,
                "visual_input_component": self.visual_input_component,
                "correct_index": correct_indices
            })


class jigsaw_puzzle_solving_painting(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/wikiart/wikiart/data/train-00000-of-00072.parquet",
        "image_output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/image-to-image_translation/jigsaw_puzzle_solving/wikiart",
        "sampling_num": 100,
        "visual_input_component": ["painting_image", "visual_mark"],
        "dataset_description": "xxx"
    }

    def parse_images_info(self):
        df = pd.read_parquet(self.anno_path)
        self.images_info = list()

        t = 0
        for _, row in enumerate(df.itertuples()):
            image = mmcv.imfrombytes(row.image["bytes"])

            if t >= 200:
                break
            t += 1
            shuffled_image, correct_indices = process_image_v3(image)
            new_image_name = self.new_image_name("jpg")
            self.exist_or_mkdir(Path(self.image_output_path))
            shuffled_image.save(Path(self.image_output_path) / new_image_name)

            self.images_info.append({
                "ori_image_path": str(Path(self.image_output_path) / new_image_name),
                "source": self.dataset_name,
                "visual_input_component": self.visual_input_component,
                "correct_index": correct_indices
            })
