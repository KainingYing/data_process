import random


def generate_incorrect_rotated_bounding_boxes(correct_boxes, image_width, image_height, num_options=3):
    """
    Generate incorrect rotated bounding boxes for a given list of correct rotated bounding boxes.
    
    :param correct_boxes: List of correct rotated bounding boxes in the format [x1, y1, x2, y2, x3, y3, x4, y4]
    :param image_width: Width of the image
    :param image_height: Height of the image
    :param num_options: Number of incorrect options to generate
    :return: List of lists containing incorrect rotated bounding boxes
    """

    incorrect_options = []

    for _ in range(num_options):
        # Copy the correct boxes to start modifications
        modified_boxes = [box.copy() for box in correct_boxes]

        for box in modified_boxes:
            # Randomly decide the type of modification
            modification_type = random.choice(["resize", "reposition", "None"])

            if modification_type == "resize":
                # Resize the box by a random factor for each point
                resize_factors = [random.uniform(0.8, 1.2) for _ in range(8)]
                box = [max(0, min(int(coord * factor), image_width if i % 2 == 0 else image_height))
                       for i, (coord, factor) in enumerate(zip(box, resize_factors))]

            elif modification_type == "reposition":
                # Reposition the box by a small random offset for each point
                offsets = [random.randint(-20, 20) for _ in range(8)]
                box = [max(0, min(coord + offset, image_width if i % 2 == 0 else image_height))
                       for i, (coord, offset) in enumerate(zip(box, offsets))]

            elif modification_type == "None":
                pass
        
        # Perform additional modifications with randomness
        additional_modifications = ["add", "remove", "duplicate"]
        random.shuffle(additional_modifications)

        for mod in additional_modifications:
            if mod == "add" and random.choice([True, False]):
                # Add a new box similar in size to existing boxes, positioned randomly
                reference_box = random.choice(correct_boxes)
                new_box = [random.randint(0, image_width), random.randint(0, image_height)] * 4  # Random points
                modified_boxes.append(new_box)

            elif mod == "remove" and modified_boxes and random.choice([True, False]):
                # Remove a random box
                modified_boxes.remove(random.choice(modified_boxes))

            elif mod == "duplicate" and random.choice([True, False]):
                # Duplicate a box with a slight offset
                duplicated_box = random.choice(correct_boxes).copy()
                offsets = [random.randint(-20, 20) for _ in range(8)]
                duplicated_box = [max(0, min(coord + offset, image_width if i % 2 == 0 else image_height))
                                  for i, (coord, offset) in enumerate(zip(duplicated_box, offsets))]
                modified_boxes.append(duplicated_box)

        # Add the modified list of boxes to the incorrect options
        incorrect_options.append(modified_boxes)

    return incorrect_options


def generate_incorrect_bounding_box_from_single_bbox(correct_box, image_width, image_height, num_options=3):
    """
    Generate incorrect bounding boxes for a given correct bounding box.
    
    :param correct_box: A correct bounding box in the format [x, y, w, h]
    :param image_width: Width of the image
    :param image_height: Height of the image
    :param num_options: Number of incorrect options to generate
    :return: List containing incorrect bounding boxes
    """

    incorrect_options = []

    for _ in range(num_options):
        # Copy the correct box to start modifications
        box = correct_box.copy()

        # Randomly decide the type of modification
        modification_type = random.choice(["resize", "reposition", "new"])

        if modification_type == "resize":
            # Resize the box by a random factor
            resize_factor_w = random.uniform(0.8, 1.2)
            resize_factor_h = random.uniform(0.8, 1.2)
            box[2] = int(box[2] * resize_factor_w)  # Modify width
            box[3] = int(box[3] * resize_factor_h)  # Modify height

        elif modification_type == "reposition":
            # Reposition the box by a small random offset
            offset_x = random.randint(-box[2] // 2, box[2] // 2)
            offset_y = random.randint(-box[3] // 2, box[3] // 2)
            box[0] = max(0, min(box[0] + offset_x, image_width - box[2]))  # Modify x
            box[1] = max(0, min(box[1] + offset_y, image_height - box[3]))  # Modify y

        elif modification_type == "new":
            # Generate a completely new box
            new_box_w = random.randint(10, image_width // 2)
            new_box_h = random.randint(10, image_height // 2)
            new_box_x = random.randint(0, image_width - new_box_w)
            new_box_y = random.randint(0, image_height - new_box_h)
            box = [new_box_x, new_box_y, new_box_w, new_box_h]

        # Add the modified box to the incorrect options
        incorrect_options.append(box)

        assert box != correct_box  # Ensure the modified box is not the same as the correct one

    return incorrect_options


def generate_incorrect_bounding_boxes(correct_boxes, image_width, image_height, num_options=3):
    """
    Generate incorrect bounding boxes for a given list of correct bounding boxes.
    
    :param correct_boxes: List of correct bounding boxes in the format [x, y, w, h]
    :param image_width: Width of the image
    :param image_height: Height of the image
    :param num_options: Number of incorrect options to generate
    :return: List of lists containing incorrect bounding boxes
    """

    incorrect_options = []

    for _ in range(num_options):
        # Copy the correct boxes to start modifications
        modified_boxes = [box.copy() for box in correct_boxes]

        for box in modified_boxes:
            # Randomly decide the type of modification
            modification_type = random.choice(["resize", "reposition", "None"])

            if modification_type == "resize":
                # Resize the box by a random factor
                resize_factor_w = random.uniform(0.8, 1.2)
                resize_factor_h = random.uniform(0.8, 1.2)
                box[2] = int(box[2] * resize_factor_w)  # Modify width
                box[3] = int(box[3] * resize_factor_h)  # Modify height

            elif modification_type == "reposition":
                # Reposition the box by a small random offset, considering the size of the box
                offset_x = random.randint(-box[2], box[2])
                offset_y = random.randint(-box[3], box[3])
                box[0] = max(0, min(box[0] + offset_x, image_width - box[2]))  # Modify x
                box[1] = max(0, min(box[1] + offset_y, image_height - box[3]))  # Modify y
            
            elif modification_type == "None":
                pass
        
        # Perform additional modifications with randomness
        additional_modifications = ["add", "remove", "duplicate"]
        random.shuffle(additional_modifications)

        for mod in additional_modifications:
            if mod == "add" and random.choice([True, False]):
                # Add a new box similar in size to existing boxes, positioned randomly
                reference_box = random.choice(correct_boxes)
                new_box_w = int(reference_box[2] * random.uniform(0.8, 1.2))
                new_box_h = int(reference_box[3] * random.uniform(0.8, 1.2))
                new_box_x = random.randint(0, max(0, image_width - new_box_w))
                new_box_y = random.randint(0, max(0, image_height - new_box_h))
                new_box = [new_box_x, new_box_y, new_box_w, new_box_h]
                modified_boxes.append(new_box)

            elif mod == "remove" and modified_boxes and random.choice([True, False]):
                # Remove a random box
                modified_boxes.remove(random.choice(modified_boxes))

            elif mod == "duplicate" and random.choice([True, False]):
                # Duplicate a box with a slight offset
                duplicated_box = random.choice(correct_boxes).copy()
                offset_x = random.randint(-duplicated_box[2] // 10, duplicated_box[2] // 10)
                offset_y = random.randint(-duplicated_box[3] // 10, duplicated_box[3] // 10)
                duplicated_box[0] = max(0, min(duplicated_box[0] + offset_x, image_width - duplicated_box[2]))
                duplicated_box[1] = max(0, min(duplicated_box[1] + offset_y, image_height - duplicated_box[3]))
                modified_boxes.append(duplicated_box)

        # Add the modified list of boxes to the incorrect options
        incorrect_options.append(modified_boxes)

        assert len(modified_boxes) > 0

    return incorrect_options

import random

def generate_incorrect_bounding_boxes_with_labels(correct_boxes, labels, image_width, image_height, num_options=3):
    """
    Generate incorrect bounding boxes for a given list of correct bounding boxes, along with their labels.
    
    :param correct_boxes: List of correct bounding boxes in the format [x, y, w, h]
    :param labels: List of labels corresponding to each bounding box
    :param image_width: Width of the image
    :param image_height: Height of the image
    :param num_options: Number of incorrect options to generate
    :return: List of tuples containing incorrect bounding boxes and their labels
    """

    incorrect_options = []

    for _ in range(num_options):
        # Copy the correct boxes and their labels to start modifications
        modified_boxes = [box.copy() for box in correct_boxes]
        modified_labels = labels.copy()

        for idx, box in enumerate(modified_boxes):
            # Randomly decide the type of modification
            modification_type = random.choice(["resize", "reposition", "None"])

            if modification_type == "resize":
                # Resize the box by a random factor
                resize_factor_w = random.uniform(0.8, 1.2)
                resize_factor_h = random.uniform(0.8, 1.2)
                box[2] = int(box[2] * resize_factor_w)  # Modify width
                box[3] = int(box[3] * resize_factor_h)  # Modify height

            elif modification_type == "reposition":
                # Reposition the box by a small random offset, considering the size of the box
                # if box[2] == 0:
                #     offset_x = 0
                # if 
                offset_x = random.randint(-box[2], box[2])
                offset_y = random.randint(-box[3], box[3])
                box[0] = max(0, min(box[0] + offset_x, image_width - box[2]))  # Modify x
                box[1] = max(0, min(box[1] + offset_y, image_height - box[3]))  # Modify y
            
            elif modification_type == "None":
                pass

        # Perform additional modifications with randomness
        additional_modifications = ["add", "remove", "duplicate"]
        random.shuffle(additional_modifications)

        for mod in additional_modifications:
            if mod == "add" and random.choice([True, False]):
                # Add a new box similar in size to existing boxes, positioned randomly, with a random label from existing ones
                reference_box = random.choice(correct_boxes)
                new_box_w = int(reference_box[2] * random.uniform(0.8, 1.2))
                new_box_h = int(reference_box[3] * random.uniform(0.8, 1.2))
                new_box_x = random.randint(0, max(0, image_width - new_box_w))
                new_box_y = random.randint(0, max(0, image_height - new_box_h))
                new_box = [new_box_x, new_box_y, new_box_w, new_box_h]
                modified_boxes.append(new_box)
                modified_labels.append(random.choice(labels))  # Choose a random label

            elif mod == "remove" and modified_boxes and random.choice([True, False]):
                # Remove a random box and its label
                remove_index = random.choice(range(len(modified_boxes)))
                del modified_boxes[remove_index]
                del modified_labels[remove_index]

            elif mod == "duplicate" and random.choice([True, False]):
                # Duplicate a box with a slight offset, keeping the same label
                duplicate_index = random.choice(range(len(modified_boxes)))
                duplicated_box = modified_boxes[duplicate_index].copy()
                offset_x = random.randint(-duplicated_box[2] // 10, duplicated_box[2] // 10)
                offset_y = random.randint(-duplicated_box[3] // 10, duplicated_box[3] // 10)
                duplicated_box[0] = max(0, min(duplicated_box[0] + offset_x, image_width - duplicated_box[2]))
                duplicated_box[1] = max(0, min(duplicated_box[1] + offset_y, image_height - duplicated_box[3]))
                modified_boxes.append(duplicated_box)
                modified_labels.append(labels[duplicate_index])  # Keep the same label

        # Add the modified list of boxes and labels to the incorrect options
        incorrect_options.append((modified_boxes, modified_labels))

        assert len(modified_boxes) == len(modified_labels) and len(modified_boxes) > 0

    return incorrect_options


import random

def generate_incorrect_rotated_bounding_boxes_with_labels(correct_boxes, labels, image_width, image_height, num_options=3):
    """
    Generate incorrect rotated bounding boxes for a given list of correct rotated bounding boxes, along with their labels.
    
    :param correct_boxes: List of correct rotated bounding boxes in the format [x1, y1, x2, y2, x3, y3, x4, y4]
    :param labels: List of labels corresponding to each bounding box
    :param image_width: Width of the image
    :param image_height: Height of the image
    :param num_options: Number of incorrect options to generate
    :return: List of tuples containing incorrect rotated bounding boxes and their labels
    """

    incorrect_options = []

    for _ in range(num_options):
        # Copy the correct boxes and their labels to start modifications
        modified_boxes = [box.copy() for box in correct_boxes]
        modified_labels = labels.copy()

        for idx, box in enumerate(modified_boxes):
            # Randomly decide the type of modification
            modification_type = random.choice(["resize", "reposition", "None"])

            if modification_type == "resize":
                # Resize the box by a random factor for each point
                resize_factors = [random.uniform(0.8, 1.2) for _ in range(8)]
                box = [max(0, min(int(coord * factor), image_width if i % 2 == 0 else image_height))
                       for i, (coord, factor) in enumerate(zip(box, resize_factors))]
                modified_boxes[idx] = box
                   

            elif modification_type == "reposition":
                # Reposition the box by a small random offset for each point
                offsets = [random.randint(-20, 20) for _ in range(8)]
                box = [max(0, min(coord + offset, image_width if i % 2 == 0 else image_height))
                       for i, (coord, offset) in enumerate(zip(box, offsets))]

                modified_boxes[idx] = box

            elif modification_type == "None":
                pass

        # Perform additional modifications with randomness
        additional_modifications = ["add", "remove", "duplicate"]
        random.shuffle(additional_modifications)

        for mod in additional_modifications:
            if mod == "add" and random.choice([True, False]):
                # Add a new box similar in size to existing boxes, positioned randomly, with a random label from existing ones
                new_box = [random.randint(0, image_width), random.randint(0, image_height)] * 4  # Random points
                modified_boxes.append(new_box)
                modified_labels.append(random.choice(labels))  # Choose a random label

            elif mod == "remove" and modified_boxes and random.choice([True, False]):
                # Remove a random box and its label
                remove_index = random.choice(range(len(modified_boxes)))
                del modified_boxes[remove_index]
                del modified_labels[remove_index]

            elif mod == "duplicate" and random.choice([True, False]):
                # Duplicate a box with a slight offset, keeping the same label
                duplicate_index = random.choice(range(len(modified_boxes)))
                duplicated_box = modified_boxes[duplicate_index].copy()
                offsets = [random.randint(-20, 20) for _ in range(8)]
                duplicated_box = [max(0, min(coord + offset, image_width if i % 2 == 0 else image_height))
                                  for i, (coord, offset) in enumerate(zip(duplicated_box, offsets))]
                modified_boxes.append(duplicated_box)
                modified_labels.append(labels[duplicate_index])  # Keep the same label

        # Add the modified list of boxes and labels to the incorrect options
        incorrect_options.append((modified_boxes, modified_labels))

        assert len(modified_boxes) == len(modified_labels) and len(modified_boxes) > 0

    return incorrect_options



    # flake8: noqa: F401, F403
import abc
import argparse
import csv
import json
import multiprocessing as mp
import numpy as np
import os, sys, time, base64, io
import os.path as osp
import copy as cp
import pickle
import random as rd
import requests
import shutil
import string
import subprocess
import warnings
import pandas as pd
from collections import OrderedDict, defaultdict
from multiprocessing import Pool, current_process
from tqdm import tqdm
from PIL import Image
from uuid import uuid4
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from tabulate import tabulate_formats, tabulate
from huggingface_hub import scan_cache_dir
import logging

def isimg(s):
    return osp.exists(s) or s.startswith('http')

def istype(s, type):
    if isinstance(s, type):
        return True
    try:
        return isinstance(eval(s), type)
    except Exception as _:
        return False

def bincount(lst):
    bins = defaultdict(lambda: 0)
    for item in lst:
        bins[item] += 1
    return bins

def read_ok(img_path):
    if not osp.exists(img_path):
        return False
    try:
        im = Image.open(img_path)
        assert im.size[0] > 0 and im.size[1] > 0
        return True
    except:
        return False

def get_cache_path(repo_id):
    hf_cache_info = scan_cache_dir()
    repos = list(hf_cache_info.repos)
    repo = None
    for r in repos:
        if r.repo_id == repo_id:
            repo = r
            break
    if repo is None:
        return None
    revs = list(repo.revisions)
    rev2keep, last_modified = None, 0
    for rev in revs:
        if rev.last_modified > last_modified:
            rev2keep, last_modified = rev, rev.last_modified 
    if rev2keep is None:
        return None
    return str(rev2keep.snapshot_path)

def md5(file_pth):
    with open(file_pth, 'rb') as f:
        hash = hashlib.new('md5')
        for chunk in iter(lambda: f.read(2**20), b''):
            hash.update(chunk)
    return str(hash.hexdigest())

def proxy_set(s):
    import os
    for key in ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
        os.environ[key] = s

logger_initialized = {}

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
    except ImportError:
        rank = 0

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    return logger

def get_rank_and_world_size():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, world_size

def circular_pred(df, extract_func=None):
    if extract_func is None:
        extract_func = lambda x: x
    df = df.sort_values('index')
    from vlmeval.utils import can_infer_option
    shift = int(1e6)

    choices = [extract_func(x) for x in df['prediction']]
    pred_map = {i: c for i, c in zip(df['index'], choices)}
    flag_map = {i: True for i in pred_map if i < 1e6}
    valid_map = {i: True for i in pred_map if i < 1e6}
    for i in df['index']:
        if i >= shift and pred_map[i] and pred_map[i - shift]:
            if pred_map[i] not in list(string.ascii_uppercase) or pred_map[i - shift] not in list(string.ascii_uppercase):
                valid_map[i % shift] = False
                continue
            if (ord(pred_map[i]) - ord(pred_map[i - shift])) % 4 == 1:
                continue
            else:
                flag_map[i % shift] = False
    flag_map = {k: v for k, v in flag_map.items() if valid_map[k]}
    flags = list(flag_map.values())
    return np.mean(flags)

def splitlen(s, sym='/'):
    return len(s.split(sym))

def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False

def d2df(D):
    return pd.DataFrame({x: [D[x]] for x in D})

def LMUDataRoot():
    if 'LMUData' in os.environ and osp.exists(os.environ['LMUData']):
        return os.environ['LMUData']
    home = osp.expanduser('~')
    root = osp.join(home, 'LMUData')
    os.makedirs(root, exist_ok=True)
    return root

def cn_string(s):
    import re
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False

try:
    import decord
except ImportError:
    pass

def build_options(option_list):
    chars = string.ascii_uppercase
    s = 'There are several options: \n'
    for c, opt in zip(chars, option_list):
        if not pd.isna(opt):
            s += f'{c}. {opt}\n'
        else:
            return s
    return s

def timestr(second=True, minute=False):
    s = datetime.datetime.now().strftime('%Y%m%d%H%M%S')[2:]
    if second:
        return s
    elif minute:
        return s[:-2]
    else:
        return s[:-4]
    
def last_modified(pth):
    stamp = osp.getmtime(pth)
    m_ti = time.ctime(stamp)
    t_obj = time.strptime(m_ti)
    t = time.strftime('%Y%m%d%H%M%S', t_obj)[2:]
    return t

def mmqa_display(question):
    question = {k.lower(): v for k, v in question.items()}
    keys = list(question.keys())
    if 'index' in keys:
        keys.remove('index')
    keys.remove('image')

    images = question['image']
    if isinstance(images, str):
        images = [images]

    idx = 'XXX'
    if 'index' in question:
        idx = question.pop('index')
    print(f'INDEX: {idx}')

    for im in images:
        image = decode_base64_to_image(im)
        w, h = image.size
        ratio = 500 / h
        image = image.resize((int(ratio * w), int(ratio * h)))
        display(image)
        
    for k in keys:
        try: 
            if not pd.isna(question[k]):
                print(f'{k.upper()}. {question[k]}')
        except ValueError:
            if False in pd.isna(question[k]):
                print(f'{k.upper()}. {question[k]}')

def encode_image_to_base64(img, target_size=-1):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    tmp = osp.join('/tmp', str(uuid4()) + '.jpg')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img.save(tmp)
    ret = encode_image_file_to_base64(tmp)
    os.remove(tmp)
    return ret

def encode_image_file_to_base64(image_path):
    if image_path.endswith('.png'):
        tmp_name = f'{timestr(second=True)}.jpg'
        img = Image.open(image_path)
        img.save(tmp_name)
        result = encode_image_file_to_base64(tmp_name)
        os.remove(tmp_name)
        return result
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        
    encoded_image = base64.b64encode(image_data)
    return encoded_image.decode('utf-8')

def decode_base64_to_image_file(base64_string, image_path):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.save(image_path)
    
def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f

def mwlines(lines, fname):
    with open(fname, 'w') as fout:
        fout.write('\n'.join(lines))

def default_set(self, args, name, default):
    if hasattr(args, name):
        val = getattr(args, name)
        setattr(self, name, val)
    else:
        setattr(self, name, default)

def dict_merge(dct, merge_dct):
    for k, _ in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):  #noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def youtube_dl(idx):
    cmd = f'youtube-dl -f best -f mp4 "{idx}"  -o {idx}.mp4'
    os.system(cmd)

def run_command(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.check_output(cmd)

def ls(dirname='.', match='', mode='all', level=1):
    if dirname == '.':
        ans = os.listdir(dirname)
    else:
        ans = [osp.join(dirname, x) for x in os.listdir(dirname)]
    assert mode in ['all', 'dir', 'file']
    assert level >= 1 and isinstance(level, int)
    if level == 1:
        ans = [x for x in ans if match in x]
        if mode == 'dir':
            ans = [x for x in ans if osp.isdir(x)]
        elif mode == 'file':
            ans = [x for x in ans if not osp.isdir(x)]
    else:
        ans = [x for x in ans if osp.isdir(x)]
        res = []
        for d in ans:
            res.extend(ls(d, match=match, mode=mode, level=level-1))
        ans = res
    return ans

def download_file(url, filename=None):
    import urllib.request

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
        
    if filename is None:
        filename = url.split('/')[-1]

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    return filename

# LOAD & DUMP
def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)

def load(f):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f) 


import random

import numpy as np


def get_padding_image_size(img_width, img_height):
    if img_height > img_width:
        pad_x0 = int((img_height - img_width) / 2)
        pad_y0 = 0
        img_width = img_height
    else:
        pad_x0 = 0
        pad_y0 = int((img_width - img_height) / 2)
        img_height = img_width
    
    return pad_x0, pad_y0, img_width, img_height


def get_bbox_coords(bbox, img_width, img_height, pad_x0, pad_y0, mode="xywh"):
    if mode == "xywh":
        x1, y1, x2, y2 = bbox[0] + pad_x0, bbox[1] + pad_y0, bbox[0] + bbox[2] + pad_x0, bbox[1] + bbox[3] + pad_y0
        x1, y1, x2, y2 = x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height
    elif mode == "xyxy":
        x1, y1, x2, y2 = bbox[0] + pad_x0, bbox[1] + pad_y0, bbox[2] + pad_x0, bbox[3] + pad_y0
        x1, y1, x2, y2 = x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height

    return x1, y1, x2, y2


def random_frame_sampling(num_frames, sampling_range, sampling_ratio):
    frame_pair_list = []
    for _step in range(1, sampling_range):
        frame_pair_list.extend(sequential_frame_sampling(num_frames, _step))
    frame_pair_list = list(set(frame_pair_list))
    sampling_num = int(len(frame_pair_list) * sampling_ratio)

    return random.sample(frame_pair_list, sampling_num)


def sequential_frame_sampling(num_frames, sampling_step):
    frame_pair_list = []
    for i in range(0, num_frames - sampling_step):
        frame_pair_list.append((i, i + sampling_step))
        frame_pair_list.append((i + sampling_step, i))
    return frame_pair_list


def mask2bbox(mask):
    # from ChatGPT4
    # 获取所有非零元素的行索引和列索引
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # 找出最小和最大的行索引和列索引
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # 返回边界框坐标
    return x_min, y_min, x_max, y_max


def replace_underscore_with_space(input_string):
    # 使用空格替换下划线
    replaced_string = input_string.replace("_", " ")
    # 将每个单词的首字母转换为小写
    return ' '.join(word.lower() for word in replaced_string.split())

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3])), 
    
    return [x1, y1, round(x2 - x1), round(y2 - y1)]


def process_points(input_points, width, height, i="f", o="n"):
    # for point in input_points:
    x, y = input_points

    if i == "n":
        # Convert from normalized to absolute coordinates if needed
        x = x * width
        y = y * height

    if o == "n":
        # Convert to normalized coordinates and round to 3 decimal places
        x = round(x / width, 3)
        y = round(y / height, 3)
    elif o == "f":
        # Round to the nearest integer for absolute coordinates
        x = round(x)
        y = round(y)
    
    return (x, y)


def process_bbox(input_bbox, width, height, i="fxyxy", o="nxyxy"):
    # Convert input_bbox to absolute coordinates (fxyxy)
    if i.startswith('n'):  # if normalized
        xmin, ymin, w_or_xmax, h_or_ymax = input_bbox
        xmin *= width
        ymin *= height
        if 'wh' in i:
            w_or_xmax *= width
            h_or_ymax *= height
        else:  # xyxy format
            w_or_xmax = w_or_xmax * width
            h_or_ymax = h_or_ymax * height
    else:  # already in full format
        xmin, ymin, w_or_xmax, h_or_ymax = input_bbox
        if 'cxcywh' in i:
            w_or_xmax = xmin + w_or_xmax  # converting to xmax
            h_or_ymax = ymin + h_or_ymax  # converting to ymax

    # Now the bbox is in fxyxy format
    if 'xywh' in o:
        width = w_or_xmax - xmin
        height = h_or_ymax - ymin
    elif 'cxcywh' in o:
        cx = (xmin + w_or_xmax) / 2
        cy = (ymin + h_or_ymax) / 2
        width = w_or_xmax - xmin
        height = h_or_ymax - ymin

    # Convert to output format
    if o.startswith('n'):  # normalize
        xmin = round(xmin / width, 3)
        ymin = round(ymin / height, 3)
        if 'xywh' in o:
            width = round(width / width, 3)
            height = round(height / height, 3)
            return [xmin, ymin, width, height]
        elif 'cxcywh' in o:
            cx = round(cx / width, 3)
            cy = round(cy / height, 3)
            width = round(width / width, 3)
            height = round(height / height, 3)
            return [cx, cy, width, height]
        else:  # nxyxy
            w_or_xmax = round(w_or_xmax / width, 3)
            h_or_ymax = round(h_or_ymax / height, 3)
            return [xmin, ymin, w_or_xmax, h_or_ymax]
    else:  # full format
        xmin = round(xmin)
        ymin = round(ymin)
        if 'xywh' in o:
            width = round(width)
            height = round(height)
            return [xmin, ymin, width, height]
        elif 'cxcywh' in o:
            cx = round(cx)
            cy = round(cy)
            width = round(width)
            height = round(height)
            return [cx, cy, width, height]
        else:  # fxyxy
            w_or_xmax = round(w_or_xmax)
            h_or_ymax = round(h_or_ymax)
            return [xmin, ymin, w_or_xmax, h_or_ymax]
