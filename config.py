max_new_tokens = 100
visualize = True
TESTS = [
    # 'atanasova_counterfactual',
    'cc_shap-posthoc',
    # 'turpin',
    # 'lanham',
    'cc_shap-cot',
]

MODELS = {
    'bakllava': 'llava-hf/bakLlava-v1-hf',
    'llava_mistral': 'llava-hf/llava-v1.6-mistral-7b-hf',
    'llava_vicuna': 'llava-hf/llava-v1.6-vicuna-7b-hf',
    'llama2-13b-chat': 'meta-llama/Llama-2-13b-chat-hf',
}


LABELS = {
    'esnli': ['A', 'B', 'C'],
    'binary': ['A', 'B'],
}

MULT_CHOICE_DATA = {
    "foil_it": ["COCO/all_images/",
                "foil-it.json"],
    # "existence": ["foil-benchmark-old/counting++/images/",
    #               "foil-benchmark/existence/existence_benchmark.test_mturk.json"],
    # "plurals": ["foil-benchmark/plurals/test_images/",
    #             "foil-benchmark/plurals/plurals_test_mturk.json"],
    # "counting_hard": ["foil-benchmark-old/counting++/images/",
    #                   "foil-benchmark/counting_hard/visual7w_counting.hard.test_mturk.json"],
    # "counting_small": ["foil-benchmark-old/counting++/images/",
    #                    "foil-benchmark/counting/visual7w_counting.small-quantities.test_mturk.json"],
    # "counting_adv": ["foil-benchmark-old/counting++/images/",
    #                  "foil-benchmark/counting_adversarial/visual7w_counting.adversarial.test_mturk.json"],
    # "relations": ["foil-benchmark/relations/test_images/",
    #               "foil-benchmark/relations/relations_test_mturk.json"],
    # "action_replace": ["foil-benchmark/actions/images_512/",
    #                    "foil-benchmark/actions/action_replace/action_replace_test_mturk.json"],
    # "actant_swap": ["foil-benchmark/actions/images_512/",
    #                 "foil-benchmark/actions/actant_swap/actant_swap_test_mturk.json"],
    # "coref": ["foil-benchmark/coref/release_too_many_is_this_in_color/images/",
    #           "foil-benchmark/coref/coref_test_visdial_train_mturk.json"],
    # "coref_hard": ["foil-benchmark/coref/release_v18/test_images/",
    #                "foil-benchmark/coref/coref_test_hard_mturk.json"],
    # "mscoco": ["COCO/all_images/", "foil-benchmark/orig_foil/foil_it_test_mturk.json"],
}

OPEN_ENDED_DATA = {
    # "vqa": ["COCO/all_images/", "VQA2.0/v2_OpenEnded_mscoco_val2014_questions.json"],
    # "gqa": ["GQA/images/", "GQA/val_all_questions.json"],
    # "gqa_balanced": ["GQA/images/", "GQA/val_balanced_questions.json"],
}
