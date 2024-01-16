import gradio as gr
import mmcv
import pandas as pd

from tqdm import tqdm

from collections import defaultdict



def get_data_list():
    # data_info = mmcv.list_from_file("/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/LMUData/taskonomy_evaluation_no_image.tsv")[1:]

    file_path = "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/data_process/LMUData/taskonomy_evaluation_no_image.tsv"  # 替换为你的文件路径
    df = pd.read_csv(file_path, sep='\t')

    data_list = defaultdict(list)

    for data_line in tqdm(df.iloc):
        question = data_line["question"]
        answer = data_line["answer"]
        A, B, C, D, E, F, G =  data_line[3: 10]

        image_path = data_line["image_path"]
        category = data_line["category"]

        data_list[category].append(
            {
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "A": A,
                "B": B,
                "C": C,
                "D": D,
                "E": E,
                "F": F,
                "G": G
            }
        )
    return data_list
        

data_list = get_data_list()


current_category = list(data_list.keys())[0]
current_index = 0

def display_data(category, index):
    global current_category, current_index
    current_category = category
    current_index = index

    category_items = data_list.get(category, [])
    if index < 0 or index >= len(category_items):
        return "Invalid index for the selected category.", None, None

    data = category_items[index]

    options_string = f"Options\n"
    if data.get("A", None) and str(data.get("A", None)) != 'nan':
        options_string += f"(A). {data['A']}\n"
    if data.get("B", None) and str(data.get("B", None)) != 'nan':
        options_string += f"(B). {data['B']}\n"
    if data.get("C", None) and str(data.get("C", None)) != 'nan':
        options_string += f"(C). {data['C']}\n"
    if data.get("D", None) and str(data.get("D", None)) != 'nan':
        options_string += f"(D). {data['D']}\n"
    if data.get("E", None) and str(data.get("E", None)) != 'nan':
        options_string += f"(E). {data['E']}\n"
    if data.get("F", None) and str(data.get("F", None)) != 'nan':
        options_string += f"(F). {data['F']}\n"
    if data.get("G", None) and str(data.get("G", None)) != 'nan':
        options_string += f"(G). {data['G']}\n"
    
    spilt_line = "=========================================================="

    show_string = f"Question: {data['question']}\n{spilt_line}\n{options_string}{spilt_line}\nAnswer: {data['answer']}"

    return show_string, data['image_path'], f'{index+1}/{len(data_list.get(category, []))}'

def next_item():
    global current_index
    category_items = data_list.get(current_category, [])
    current_index = min(current_index + 1, len(category_items) - 1)
    return display_data(current_category, current_index)

def previous_item():
    global current_index
    category_items = data_list.get(current_category, [])
    current_index = max(current_index - 1, 0)
    return display_data(current_category, current_index)

with gr.Blocks() as iface:
    with gr.Row():
        category_dropdown = gr.Dropdown(list(data_list.keys()), label="Select Category")
        prev_button = gr.Button("Previous")
        next_button = gr.Button("Next")
    index_output = gr.Textbox(label="Current Index", interactive=False)
    question_output = gr.Textbox(label="Question and Options")
    image_output = gr.Image(label="Image", width=768)
    # index_output = gr.Textbox(label="Current Index", interactive=False)

    category_dropdown.change(
        lambda category: display_data(category, 0), 
        inputs=[category_dropdown], 
        outputs=[question_output, image_output, index_output]
    )
    prev_button.click(
        previous_item, [], [question_output, image_output, index_output]
    )
    next_button.click(
        next_item, [], [question_output, image_output, index_output]
    )

# iface.launch(share=True, server_port=10067)

iface.queue(api_open=True).launch(share=True, server_name='0.0.0.0', server_port=10043)
