import os

from PIL import Image
import mmcv
from openai import OpenAI

os.environ["OPENAI_BASE_URL"] = "https://api.openai-sb.com/v1"


class BaseDataset:
    def __init__(self):
        for key, value in self.DATA_METAINFO.items():
            setattr(self, key, value)

        # self.image_dict = {"source": self.dataset_name, "visual_input_component": self.visual_input_component}
        self.parse_dataset_info()
        self.parse_images_info()

    def parse_dataset_info(self):
        self.dataset_info = dict()
        self.dataset_info["dataset_description"] = self.dataset_description
        self.dataset_info["sampling_num"] = self.sampling_num
        self.dataset_info["dataset_name"] = self.dataset_name
        self.dataset_info["visual_input_component"] = self.visual_input_component
    
    @staticmethod
    def new_image_name(e="jpg"):
        import uuid
        new_image_name = str(uuid.uuid4()) + f".{e}"
        return new_image_name
    
    @staticmethod
    def exist_or_mkdir(folder_path):
        if not os.path.exists(folder_path):
            # If the folder does not exist, create it
            os.makedirs(folder_path)
    
    @property
    def dataset_name(self):
        return self.__class__.__name__
    
    def sample(self):
        import random
        if self.sampling_num > len(self.images_info):
            self.sampling_num = len(self.images_info)
        
        output_list = random.sample(self.images_info, self.sampling_num)
        return output_list
    
    def image_dict(self, image_path, width=None, height=None):
        # if width is None or height is None:
        #     width, height = self.get_image_width_height(image_path)
        # assert os.path.exists(image_path)
        return {"source": self.dataset_name, "visual_input_component": self.visual_input_component, "width": width, "height": height}
    
    def get_image_width_height(self, image_name):
        with Image.open(image_name) as img:
            # 获取图像的宽度和高度
            width, height = img.size
        return width, height
    
    def save_image(self, image, image_path):
        img = Image.fromarray(image.astype('uint8'), 'L')
        img.save(image_path)

    def save_rgb_image(self, image, image_path):
        # cv2.imwrite(str(image_path), image)
        mmcv.imwrite(image, image_path)
    
    @staticmethod
    def openai_generate(sys, user):
        openai_client = OpenAI()
        while True:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    response_format={ "type": "json_object" },
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user}
                    ])
                output = eval(response.choices[0].message.content)
                break
            except:
                pass
        
        return output
    
    @staticmethod
    def post_process(qa_json, question=None):
        import random
        import copy
        # verify

        assert qa_json["num_wrong_choices"]
        assert qa_json["gt"] or qa_json["gt"] == 0
        assert qa_json["question"]
        assert qa_json["wrong_choices_list"]

        assert len(qa_json["wrong_choices_list"]) == qa_json["num_wrong_choices"]
        assert qa_json["gt"] not in qa_json["wrong_choices_list"]

        # prepare multi choices
        gt_index = random.randrange(0, qa_json["num_wrong_choices"] + 1)
        
        choice_list = copy.deepcopy(qa_json["wrong_choices_list"])
        random.shuffle(choice_list)
        choice_list.insert(gt_index, qa_json["gt"])

        qa_json["choice_list"] = choice_list
        qa_json["gt_index"] = gt_index

        if question is not None:
            qa_json['question'] = question
        return qa_json

    # def process_category_name(self, category):
        
