import time, sys
import torch
print("Cuda is available:", torch.cuda.is_available())
from accelerate import Accelerator
import json
from PIL import Image
import random, os
from tqdm import tqdm

from load_models import load_models
from read_datasets import read_data
from generation_and_prompting import *
from mm_shap_cc_shap import *
from other_faith_tests import *
from config import *

torch.cuda.empty_cache()
accelerator = Accelerator()
accelerator.free_memory()

from transformers.utils import logging
logging.set_verbosity_error()
import logging
logging.getLogger('shap').setLevel(logging.ERROR)

random.seed(42)

t1 = time.time()

c_task = sys.argv[1]
model_name = sys.argv[2]
save_json = int(sys.argv[3])
data_root = sys.argv[4]

model, tokenizer = load_models(model_name)

if __name__ == '__main__':
    ############################# run evals on all valse instruments
    for c_task in MULT_CHOICE_DATA.keys():
        if c_task != "mscoco":
            res_dict = {}
            formatted_samples_pairwise, formatted_samples_caption, formatted_samples_foil = [], [], []
            correct_answers, wrong_answers, image_paths = [], [], []
            acc_r, p_c, p_f = 0, 0, 0
            count = 0
            print("Preparing data...")
            # read the valse data from the json files
            images_path = f"{data_root}{MULT_CHOICE_DATA[c_task][0]}"
            foils_path = f"{data_root}{MULT_CHOICE_DATA[c_task][1]}"
            foils_data = read_data(c_task, foils_path, images_path, data_root)

            for foil_id, foil in foils_data.items():  # tqdm
                if c_task == 'mscoco':
                    # for everything other than VALSE: pretend like the sample was accepted by annotators
                    caption_fits = 3
                else: # the subtask stems from VALSE data
                    caption_fits = foil['mturk']['caption'] # take only samples accepted by annotators
                if caption_fits >= 2:  # MTURK filtering! Use only valid set
                    test_img_path = os.path.join(images_path, foil["image_file"])
                    if c_task == 'mscoco':
                        confounder = random.sample(sorted(foils_data.items()), 1)[0][1]
                        test_sentences = [foil["caption"], confounder["caption"]]
                    else:
                        if c_task == 'plurals':
                            test_sentences = [foil["caption"][0], foil["foil"][0]]
                        else:
                            test_sentences = [foil["caption"], foil["foil"][0]]

                    # shuffle the order of caption and foil such that the correct answer is not always A
                    if random.choice([0, 1]) == 0:
                        formatted_sample_pairwise = format_example_valse_pairwise(test_sentences[0], test_sentences[1])
                        correct_answer, wrong_answer = 'A', 'B'
                    else:
                        formatted_sample_pairwise = format_example_valse_pairwise(test_sentences[1], test_sentences[0])
                        correct_answer, wrong_answer = 'B', 'A'

                    formatted_sample_caption = format_example_valse(test_sentences[0])
                    formatted_sample_foil = format_example_valse(test_sentences[1])

                    formatted_samples_pairwise.append(formatted_sample_pairwise)
                    formatted_samples_caption.append(formatted_sample_caption)
                    formatted_samples_foil.append(formatted_sample_foil)
                    correct_answers.append(correct_answer)
                    wrong_answers.append(wrong_answer)
                    image_paths.append(test_img_path)

                    count += 1
           
            print("Done preparing data. Running test...")
            for k, formatted_sample_pairwise, formatted_sample_caption, formatted_sample_foil, correct_answer, wrong_answer, image_path in zip(range(len(formatted_samples_pairwise)), formatted_samples_pairwise, formatted_samples_caption, formatted_samples_foil, correct_answers, wrong_answers, image_paths): # tqdm
                raw_image = Image.open(image_path) # read image
                if c_task in MULT_CHOICE_DATA.keys():
                    labels = LABELS['binary']
                elif c_task in OPEN_ENDED_DATA.keys():
                    labels = None
                else:
                    labels = LABELS[c_task]
                # compute model accuracy post-hoc
                inp_ask_for_prediction = prompt_answer_with_input(formatted_sample_pairwise, c_task)
                prediction = vlm_predict(inp_ask_for_prediction, raw_image, model, tokenizer, c_task, labels=labels)
                acc_r_sample = evaluate_prediction(prediction, correct_answer, c_task)
                acc_r += acc_r_sample

                inp_ask_for_prediction = prompt_answer_with_input(formatted_sample_caption, c_task)
                prediction = vlm_predict(inp_ask_for_prediction, raw_image, model, tokenizer, c_task, labels=labels)
                p_c_sample = evaluate_prediction(prediction, 'A', c_task)
                p_c += p_c_sample

                inp_ask_for_prediction = prompt_answer_with_input(formatted_sample_foil, c_task)
                prediction = vlm_predict(inp_ask_for_prediction, raw_image, model, tokenizer, c_task, labels=labels)
                p_f_sample = evaluate_prediction(prediction, 'B', c_task)
                p_f += p_f_sample


                res_dict[f"{c_task}_{model_name}_{k}"] = {
                    "image_path": image_path,
                    "sample": formatted_sample_pairwise,
                    "correct_answer": correct_answer,
                    "post-hoc": {
                        "acc_r": acc_r_sample,
                        "p_c": p_c_sample,
                        "p_f": p_f_sample,
                    },
                }

            if save_json:
                # save results to a json file, make results_json directory if it does not exist
                if not os.path.exists('results_json_valse'):
                    os.makedirs('results_json_valse')
                with open(f"results_json_valse/{c_task}_{model_name}_{count}_valse_eval.json", 'w') as file:
                    json.dump(res_dict, file)


            print(f"Ran valse eval on {c_task} {count} samples with model {model_name}. Reporting accuracy metrics.\n")
            print(f"acc_r %          : {acc_r*100/count:.2f}  ")
            print(f"p_c %            : {p_c*100/count:.2f}  ")
            print(f"p_f %            : {p_f*100/count:.2f}  ")


    c = time.time()-t1
    print(f"\nThis script ran for {c // 86400:.2f} days, {c // 3600 % 24:.2f} hours, {c // 60 % 60:.2f} minutes, {c % 60:.2f} seconds.")
