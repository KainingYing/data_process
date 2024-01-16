single_choice_classification_sys_prompt = "I need you to help me create options for a classification problem with single choice, where the answer is a category. Additionally, I will provide you with a category space for creating other options. Formally, I will provide you with a category space, a ground truth category, a question, and you will assist me in completing a dictionary and returning it as JSON. The output dict must contain num_choices,question,gt_category,gt_choice,choice_a,choice_b...(according to the number of the num_choices)"

multi_object_detection_with_category_sys_prompt = "I need you to help me create options for a multiple object detection problem, where the answer is a multiple category-bounding box coordinates pairs. Additionally, I will provide you with a category space for creating other options. Formally, I will provide you with a category space, a ground truth, a question (you can use this question directly when you return), and you will assist me in completing a dictionary and returning it as JSON."

multi_object_detection_without_category_sys_prompt = "I need you to help me create options for a multiple object detection problem, where the answer is a multiple bounding box coordinates (xyxy format). Additionally, I will provide you with GT bounding box coordinates for creating other options. Please create other incorrect options based on the GT bounding box coordinates I provided, where the incorrect options may include offsetting the bounding box, missing the bounding box, adding extra bounding boxes, and so on. Formally, I will provide you with ground truth multiple bounding box coordinates, a question (you can use this question directly when you return), and you will assist me in completing a dictionary and returning it as JSON."

