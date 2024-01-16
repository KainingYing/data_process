import argparse

import mmcv
import torchshow as ts

import numpy as np


def show_bbox(image_path, bbox_list, format):
    bbox_list = eval(bbox_list)
    if format == "xyxy":
        # bbox_list = eval(bbox_list)
        pass
    else:
        bbox_list = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bbox_list]
    
    img = mmcv.imshow_bboxes(image_path, np.array(bbox_list), show=False)
    ts.save(img)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Example script to parse arguments.")
    parser.add_argument('image', type=str,)
    parser.add_argument('bbox', type=str)
    parser.add_argument('format', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    show_bbox(args.image, args.bbox, args.format)
