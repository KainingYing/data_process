dataset = {
    # ==============================================================================================================================
    # low_level_vision
    "depth_estimation": {
        "dataset_list": [
            "taskonomy",
            "nyu",
            "nuscenes",
            "kitti"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/low_level_vision/shape_recognition"
    },
    "height_estimation": {
        "dataset_list": [
            "gta_height",
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/low_level_vision/height_depth_estimation"
    },
    # ==============================================================================================================================
    # visual_recognition
    "shape_recognition": {
        "dataset_list": [
            # "twod_geometric_shapes_dataset",
            "gpt_auto_generated_shape"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/shape_recognition"
    },
    "color_recognition": {
        "dataset_list": [
            "python_auto_generated_color"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/color_recognition"
    },
    "texture_material_recognition": {
        "dataset_list": [
            "kth",
            "kyberge",
            "uiuc",
            "opensurfaces"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/texture_material_recognition"
    },
    "painting_recognition": {
        "dataset_list": [
            "wikiart",
            "best_artwork_of_all_time",
            "van_gogh_paintings_dataset",
            "chinese_patinting_internet"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/painting_recognition"
    },
    "sculpture_recognition": {
        "dataset_list": [
            "sculpture_internet",
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/sculpture_recognition"
    },
    "logo_and_brand_recognition": {
        "dataset_list": [
            "fake_real_logo_detection_dataset",
            "flickr_sport_logos_10",
            "car_logos_dataset",
            "logo627"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/logo_and_brand_recognition"
    },
    "season_recognition": {
        "dataset_list": [
            "image_season_recognition"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/image_season_recognition"
    },
    "film_and_television_recognition": {
        "dataset_list": [
            "internet_poster",
            "movie_posters_kaggle"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/film_and_television_recognition"
    },
    "scene_recognition": {
        "dataset_list": [
            # "indoor_scene_recognition",
            "places365"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/scene_recognition"
    },
    "landmark_recognition": {
        "dataset_list": [
            "landmark_internet",
            # "places365"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/landmark_recognition"
    },
    "national_flag_recognition": {
        "dataset_list": [
            "country_flag"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/national_flag_recognition"
    },
    "fashion_recognition": {
        "dataset_list": [
            "fashion_mnist",
            "deepfashion"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/fashion_recognition"
    },
    "abstract_visual_recognition": {
        "dataset_list": [
            # "quickdraw",
            "imagenet_sketch"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_recognition/abstract_visual_recognition"
    },
    # ==============================================================================================================================
    # localization
    "small_object_detection": {
        "dataset_list": [
            "sod4bird",
            # "drone2021",
            "tinyperson"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/localization/small_object_detection"
    },
    "rotated_object_detection": {
        "dataset_list": [
            # "dota",
            "ssdd_inshore",
            "ssdd_offshore"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/localization/rotated_object_detection"
    },
    # ==============================================================================================================================
    # pixel-level perception
    "polygon_localization": {
        "dataset_list": [
            # "coco_polygon",
            # "youtubevis2019_polygon",
            "ovis_polygon"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/pixel_level_perception/image_matting"
    },
    "image_matting": {
        "dataset_list": [
            "am2k",
            "aim500"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/pixel_level_perception/image_matting"
    },
    # ==============================================================================================================================
    # ocr
    "handwritten_text_recognition": {
        "dataset_list": [
            "iam_line",
            "iam_page"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/ocr/handwritten_text_recognition"
    },
    "handwritten_mathematical_expression_recognition": {
        "dataset_list": [
            "hme100k",
            "crohme2014"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/ocr/handwritten_mathematical_expression_recognition"
    },
    # ==============================================================================================================================
    # visual_prompt_understanding
    "visual_prompt_understanding": {
        "dataset_list": [
            "vipbench"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_prompt_understanding"
    },
    "som_recognition": {
        "dataset_list": [
            "sombench_flickr30k_grounding",
            "sombench_refcocog_refseg"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/som_recognition"
    },
    # ==============================================================================================================================
    # image2image_translation
    "jigsaw_puzzle_solving": {
        "dataset_list": [
            "jigsaw_puzzle_solving_natural",
            "jigsaw_puzzle_solving_painting"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/image-to-image_translation/jigsaw_puzzle_solving"
    },
    # ==============================================================================================================================
    # relation reasoning
    "social_relation_recognition": {
        "dataset_list": [
            "social_relation_dataset",
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/relation_reasoning/social_relation_recognition",
    },
    "human_object_interaction_recognition": {
        "dataset_list": [
            "hicodet",
            # "vcoco"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/relation_reasoning/human_object_interaction_recognition",
    },
    "human_interaction_understanding": {
        "dataset_list": [
            # "hicodet_hiu",
            "bit"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/relation_reasoning/human_interaction_understanding",
    },
    # ==============================================================================================================================
    # visual_illusion
    "color_assimilation": {
        "dataset_list": [
            "gvil_assimilation"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/color_assimilation",
    },
    "color_constancy": {
        "dataset_list": [
            "gvil_constancy"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/color_constancy",
    },
    "color_contrast": {
        "dataset_list": [
            "gvil_contrast"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/color_contrast",
    },
    "geometrical_perspective": {
        "dataset_list": [
            "gvil_perspective"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/geometrical_perspective",
    },
    "geometrical_relativity": {
        "dataset_list": [
            "gvil_relativity"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/geometrical_relativity",
    },
    # ==============================================================================================================================
    # visual_coding
    "eqn2latex": {
        "dataset_list": [
            "im2latex90k"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/eqn2latex",
    },
    "screenshot2code": {
        "dataset_list": [
            "pix2code_andriod",
            "pix2code_ios",
            "pix2code_web"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/screenshot2code",
    },
    "sketch2code": {
        "dataset_list": [
            "sketch2code_kaggle"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/visual_illusion/sketch2code",
    },
    # ==============================================================================================================================
    # counting
    "counting_by_category": {
        "dataset_list": [
            "fsc147_category",
            "countqa_vqa",
            "countqa_cocoqa",
            "tallyqa_simple"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/counting/counting_by_category",
    },
    "counting_by_reasoning": {
        "dataset_list": [
            "tallyqa_complex",
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/counting/counting_by_reasoning",
    },
    "counting_by_visual_prompting": {
        "dataset_list": [
            "fsc147"
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/counting/counting_by_visual_prompting",
    },
    # ==============================================================================================================================
    # doc_understanding
    "visual_document_information_extraction": {
        "dataset_list": [
            "funsd",
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/doc_understanding/visual_document_information_extraction",
    },
    "table_structure_recognition": {
        "dataset_list": [
            "scitsr",
        ],
        "output_path": "/mnt/petrelfs/share_data/yingkaining/lvlm_evaluation/taskonomy_evaluation_data/doc_understanding/table_structure_recognition",
    },
}
