import os
import argparse
import warnings
from pathlib import Path
from PIL import Image
import base64
import datetime
import string
from uuid import uuid4
import os.path as osp
warnings.filterwarnings("ignore")

import mmcv
from tqdm import tqdm
import pandas as pd


TASK_NAME_LIST = []
qa_path = "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/taskonomy_evaluation_data"
for task_name in Path(qa_path).iterdir():
    for sub_task_name in Path(task_name).iterdir():
        if (sub_task_name / "qa_info").exists():
            TASK_NAME_LIST.append(sub_task_name.stem)

print(TASK_NAME_LIST)
    

def timestr(second=True, minute=False):
    s = datetime.datetime.now().strftime('%Y%m%d%H%M%S')[2:]
    if second:
        return s
    elif minute:
        return s[:-2]
    else:
        return s[:-4]


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
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(tmp_name)
        result = encode_image_file_to_base64(tmp_name)
        os.remove(tmp_name)
        return result
    
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
    
    encoded_image = base64.b64encode(image_data)
    return encoded_image.decode('utf-8')


def main(args):
    dataset_config_file = args.dataset_config

    dataset_config_file = mmcv.Config.fromfile(dataset_config_file)

    i = 0
    data = {
        "index": [],
        "question": [],
        "answer": [],
        "A": [],
        "B": [],
        "C": [],
        "D": [],
        "E": [],
        "F": [],
        "G": [],
        "image_path": [],
        "image": [],
        "category": []
    }
    
    for task_name in tqdm(TASK_NAME_LIST):
        if "flag" in task_name:
            continue
        dataset_config = dataset_config_file.dataset[task_name]
        data_temp = {
            "index": [],
            "question": [],
            "answer": [],
            "A": [],
            "B": [],
            "C": [],
            "D": [],
            "E": [],
            "F": [],
            "G": [],
            "image_path": [],
            "image": [],
            "category": []
        }

        for qa_json in sorted((Path(dataset_config["output_path"]) / "qa_info").iterdir()):
            try:
                qa_info = mmcv.load(qa_json)

                index = i
                question = qa_info["question"]

                A = qa_info["choice_list"][0] if len(qa_info["choice_list"]) > 0 else None
                B = qa_info["choice_list"][1] if len(qa_info["choice_list"]) > 1 else None
                C = qa_info["choice_list"][2] if len(qa_info["choice_list"]) > 2 else None
                D = qa_info["choice_list"][3] if len(qa_info["choice_list"]) > 3 else None
                E = qa_info["choice_list"][4] if len(qa_info["choice_list"]) > 4 else None
                F = qa_info["choice_list"][5] if len(qa_info["choice_list"]) > 5 else None
                G = qa_info["choice_list"][6] if len(qa_info["choice_list"]) > 6 else None

                answer = string.ascii_uppercase[qa_info["gt_index"]]
                img = Image.open(qa_info['original_image_path'])
                image = encode_image_to_base64(img)
                
                image_path = qa_info['original_image_path']
            except:
                continue

            data["index"].append(index)
            data["A"].append(A)
            data["B"].append(B)
            data["C"].append(C)
            data["D"].append(D)
            data["E"].append(E)
            data["F"].append(F)
            data["G"].append(G)
            data["answer"].append(answer)
            data["question"].append(question)
            data["image_path"].append(image_path)
            data["image"].append(image)
            data["category"].append(task_name)

            data_temp["index"].append(index)
            data_temp["A"].append(A)
            data_temp["B"].append(B)
            data_temp["C"].append(C)
            data_temp["D"].append(D)
            data_temp["E"].append(E)
            data_temp["F"].append(F)
            data_temp["G"].append(G)
            data_temp["answer"].append(answer)
            data_temp["question"].append(question)
            data_temp["image_path"].append(image_path)
            data_temp["image"].append(image)
            data_temp["category"].append(task_name)

            i += 1
        df = pd.DataFrame(data_temp)
        df.to_csv(Path('LMUData') / 'temp' / f'{task_name}.tsv', sep='\t', index=False)
            
    df = pd.DataFrame(data)
    df.to_csv(Path('LMUData') / 'taskonomy_evaluation.tsv', sep='\t', index=False)
    
    del data["image"]
    df = pd.DataFrame(data)
    df.to_csv(Path('LMUData') / 'taskonomy_evaluation_no_image.tsv', sep='\t', index=False)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Example script to parse arguments.")
    parser.add_argument('--dataset_config', type=str, default="/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/dataset_config.py", help='The path of dataset config.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
