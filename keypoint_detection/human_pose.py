import mmcv
import uuid
import sys
sys.path.append("data_process")

from base_dataset import BaseDataset


class mscoco(BaseDataset):
    def __init__(self) -> None:
        dataset_config = mmcv.Config.fromfile("/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/dataset_config.py")
        dataset_config = dataset_config.keypoint_detection["mscoco"]

        self.dataset_name = dataset_config["dataset_name"]
        self.dataset_path = dataset_config["data_path"]

        self.data_info = self.get_data_info()

    def get_data_info(self):
        pass


if __name__ == '__main__':
    a = mscoco()


