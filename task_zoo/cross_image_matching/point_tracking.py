from pathlib import Path
from collections import OrderedDict

from base_dataset import BaseDataset
import mmcv

from prompt.utils import *

class tapvid_davis(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/data/point_tracking/tap_vid/tapvid_davis/tapvid_davis.pkl",
        "save_image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/taskonomy_evaluation_data/cross_image_matching/point_tracking/images",
        "sampling_num": 100,
        "url": "internet",
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx",
        "sampling_range": 50,
        "sampling_ratio": 0.01,

    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info = mmcv.load(self.anno_path)

        for _, video_info in tqdm(anno_info.items()):
            num_frames = video_info["video"].shape[0]
            num_points = video_info["points"].shape[0]

            sampling_pair_list = random_frame_sampling(num_frames=num_frames, sampling_range=self.sampling_range, sampling_ratio=self.sampling_ratio)

            for sampling_pair in sampling_pair_list:
                point_id = random.randint(0, num_points-1)
                point_1 = video_info["points"][point_id][sampling_pair[0]]
                point_2 = video_info["points"][point_id][sampling_pair[1]]
            
                image_1 = video_info["video"][sampling_pair[0]]
                image_2 = video_info["video"][sampling_pair[1]]

                out_image_name_1 = os.path.join(self.save_image_path, self.new_image_name())
                out_image_name_2 = os.path.join(self.save_image_path, self.new_image_name())
                self.save_rgb_image(image_1, out_image_name_1)
                self.save_rgb_image(image_2, out_image_name_2)

                image_info = self.image_dict(out_image_name_1)

                image_info.update(
                    {
                        "original_image_path": [out_image_name_1, out_image_name_2],
                        "point1": point_1.tolist(),
                        "point2": point_2.tolist()
                    }
                )
            
                self.images_info.append(image_info)


class tapvid_rgb_stacking(BaseDataset):
    DATA_METAINFO = {
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/data/point_tracking/tap_vid/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl",
        "save_image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/taskonomy_evaluation_data/cross_image_matching/point_tracking/images",
        "sampling_num": 100,
        "url": "internet",
        "visual_input_component": ["synthetic_image"],
        "dataset_description": "xxx",
        "sampling_range": 50,
        "sampling_ratio": 0.0001,
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        self.images_info = list()
        anno_info = mmcv.load(self.anno_path)

        for video_info in tqdm(anno_info):
            num_frames = video_info["video"].shape[0]
            num_points = video_info["points"].shape[0]

            sampling_pair_list = random_frame_sampling(num_frames=num_frames, sampling_range=self.sampling_range, sampling_ratio=self.sampling_ratio)

            for sampling_pair in sampling_pair_list:
                point_id = random.randint(0, num_points-1)
                point_1 = video_info["points"][point_id][sampling_pair[0]]
                point_2 = video_info["points"][point_id][sampling_pair[1]]
            
                image_1 = video_info["video"][sampling_pair[0]]
                image_2 = video_info["video"][sampling_pair[1]]

                out_image_name_1 = os.path.join(self.save_image_path, self.new_image_name())
                out_image_name_2 = os.path.join(self.save_image_path, self.new_image_name())
                self.save_rgb_image(image_1, out_image_name_1)
                self.save_rgb_image(image_2, out_image_name_2)

                image_info = self.image_dict(out_image_name_1)

                image_info.update(
                    {
                        "original_image_path": [out_image_name_1, out_image_name_2],
                        "point1": point_1.tolist(),
                        "point2": point_2.tolist()
                    }
                )
            
                self.images_info.append(image_info)
