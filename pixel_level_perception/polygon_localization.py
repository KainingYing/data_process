from pathlib import Path


from base_dataset import BaseDataset


class coco_polygon(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/coco/val2017",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/coco/annotations/instances_val2017.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        from collections import defaultdict

        self.images_info = list()
        self.category_space = []
        anno_data_info = mmcv.load(self.anno_path)
        
        image2anno = defaultdict(list)

        for anno_info in anno_data_info["annotations"]:
            # segmentation = anno_info["segmentation"]
            image2anno[anno_info["image_id"]].append(anno_info)

        id2category = dict()
        for cate_info in anno_data_info["categories"]:
            id2category[cate_info['id']] = cate_info["name"]
            self.category_space.append(cate_info["name"])
        
        for image_info in anno_data_info["images"]:
            segment_list = image2anno[image_info["id"]]

            category_list = []
            segmentation_lsit = []
            for segment_info in segment_list:
                category = id2category[segment_info["category_id"]]
                segmentation = segment_info["segmentation"]

                category_list.append(category)
                segmentation_lsit.append(segmentation)
            
            image_path = Path(self.image_path) / image_info["file_name"]
            
            width, height = self.get_image_width_height(image_path)
            
            image_info = self.image_dict
            image_info.update(
                {
                    "original_image_path": str(image_path),
                    "width": width,
                    "height": height,
                    "segmentation_list": segmentation_lsit,
                    "category": category_list
                }
            )

            self.images_info.append(image_info)

        self.dataset_info["category_space"] = self.category_space


class youtubevis2019_polygon(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ytvis_2019/train/JPEGImages",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/ytvis_2019/train.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        from collections import defaultdict

        self.images_info = list()
        self.category_space = []
        anno_data_info = mmcv.load(self.anno_path)
        
        video2anno = defaultdict(list)

        for anno_info in anno_data_info["annotations"]:
            # segmentation = anno_info["segmentation"]
            video2anno[anno_info["video_id"]].append(anno_info)

        id2category = dict()
        for cate_info in anno_data_info["categories"]:
            id2category[cate_info['id']] = cate_info["name"]
            self.category_space.append(cate_info["name"])
        
        for video_info in anno_data_info["videos"]:
            segment_list = video2anno[video_info["id"]]

            for i, file_name in enumerate(video_info["file_names"]):
                image_path = Path(self.image_path) / file_name

                segmentation_list = []
                category_list = []

                for segmentation in segment_list:
                    if segmentation["segmentations"][i] == None:
                        continue
                    else:
                        category_list.append(id2category[segmentation["category_id"]])
                        segmentation_list.append(segmentation["segmentations"][i])

                if len(category_list) == 0:
                    continue
                width, height = self.get_image_width_height(image_path)
                
                image_info = self.image_dict
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "width": width,
                        "height": height,
                        "segmentation_list": segmentation_list,
                        "category": category_list
                    }
                )

                self.images_info.append(image_info)

        self.dataset_info["category_space"] = self.category_space


class ovis_polygon(BaseDataset):
    DATA_METAINFO = {
        "image_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/OpenDataLab___OVIS/raw/train",
        "anno_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data/OpenDataLab___OVIS/raw/annotations_train.json",
        "sampling_num": 200,
        "visual_input_component": ["natural_image"],
        "dataset_description": "xxx"
    }
    
    def parse_dataset_info(self):
        super().parse_dataset_info()
    
    def parse_images_info(self):
        import mmcv
        from collections import defaultdict

        self.images_info = list()
        self.category_space = []
        anno_data_info = mmcv.load(self.anno_path)
        
        video2anno = defaultdict(list)

        for anno_info in anno_data_info["annotations"]:
            # segmentation = anno_info["segmentation"]
            video2anno[anno_info["video_id"]].append(anno_info)

        id2category = dict()
        for cate_info in anno_data_info["categories"]:
            id2category[cate_info['id']] = cate_info["name"]
            self.category_space.append(cate_info["name"])
        
        for video_info in anno_data_info["videos"]:
            segment_list = video2anno[video_info["id"]]

            for i, file_name in enumerate(video_info["file_names"]):
                image_path = Path(self.image_path) / file_name

                segmentation_list = []
                category_list = []

                for segmentation in segment_list:
                    if segmentation["segmentations"][i] == None:
                        continue
                    else:
                        category_list.append(id2category[segmentation["category_id"]])
                        segmentation_list.append(segmentation["segmentations"][i])

                if len(category_list) == 0:
                    continue
                width, height = self.get_image_width_height(image_path)
                
                image_info = self.image_dict
                image_info.update(
                    {
                        "original_image_path": str(image_path),
                        "width": width,
                        "height": height,
                        "segmentation_list": segmentation_list,
                        "category": category_list
                    }
                )

                self.images_info.append(image_info)

        self.dataset_info["category_space"] = self.category_space