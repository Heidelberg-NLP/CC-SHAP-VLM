import time, sys
import torch
print("Cuda is available:", torch.cuda.is_available())
from accelerate import Accelerator
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
num_samples = int(sys.argv[3])
data_root = sys.argv[4]

model, tokenizer = load_models(model_name)

############################# run experiments on data
res_dict = {}
count, t_shap_c_sum, t_shap_f_sum = 0, 0, 0
p_c, p_f = 0, 0
image_paths, formatted_samples_pairwise, formatted_samples_caption, formatted_samples_foil = [], [], [], []

print("Preparing data...")
if c_task in MULT_CHOICE_DATA.keys():     ###### VALSE tests
    # read the valse data from the json files
    images_path = f"{data_root}{MULT_CHOICE_DATA[c_task][0]}"
    foils_path = f"{data_root}{MULT_CHOICE_DATA[c_task][1]}"
    foils_data = read_data(c_task, foils_path, images_path, data_root)

    for foil_id, foil in tqdm(foils_data.items()):  # tqdm
        if count + 1 > num_samples:
            break
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

            formatted_sample_caption = format_example_valse(test_sentences[0])
            formatted_sample_foil = format_example_valse(test_sentences[1])

            formatted_samples_caption.append(formatted_sample_caption)
            formatted_samples_foil.append(formatted_sample_foil)
            image_paths.append(test_img_path)

            count += 1
else:
    raise NotImplementedError(f'Your specified task has no implementation: {c_task}')

print("Done preparing data. Running test...")
for k, formatted_sample_caption, formatted_sample_foil, image_path in tqdm(zip(range(len(formatted_samples_caption)), formatted_samples_caption, formatted_samples_foil, image_paths)):
    raw_image = Image.open(image_path).convert("RGB") # read image
    if c_task in MULT_CHOICE_DATA.keys():
        labels = LABELS['binary']
    elif c_task in OPEN_ENDED_DATA.keys():
        labels = None
    else:
        labels = LABELS[c_task]
    # compute model accuracy post-hoc
    inp_ask_for_prediction = prompt_answer_with_input(formatted_sample_caption, c_task)
    prediction = vlm_predict(copy.deepcopy(inp_ask_for_prediction), raw_image, model, tokenizer, c_task, labels=labels)
    accuracy_sample = evaluate_prediction(prediction, 'A', c_task)
    p_c += accuracy_sample
    _, mm_score, _, _ = explain_VLM(inp_ask_for_prediction, raw_image, model, tokenizer, max_new_tokens=1)
    t_shap_c_sum += mm_score

    inp_ask_for_prediction = prompt_answer_with_input(formatted_sample_foil, c_task)
    prediction = vlm_predict(copy.deepcopy(inp_ask_for_prediction), raw_image, model, tokenizer, c_task, labels=labels)
    accuracy_sample = evaluate_prediction(prediction, 'B', c_task)
    p_f += accuracy_sample
    _, mm_score, _, _ = explain_VLM(inp_ask_for_prediction, raw_image, model, tokenizer, max_new_tokens=1)
    t_shap_f_sum += mm_score

print(f"Ran MM-SHAP non-pairwise on {c_task} {count} samples with model {model_name}. Reporting accuracy and faithfulness percentage.\n")
print(f"p_c %          : {p_c*100/count:.2f}  ")
print(f"p_f %          : {p_f*100/count:.2f}  ")
print(f"T-SHAP_c %     : {t_shap_c_sum/count*100:.2f}  ")
print(f"T-SHAP_f %     : {t_shap_f_sum/count*100:.2f}  ")


c = time.time()-t1
print(f"\nThis script ran for {c // 86400:.2f} days, {c // 3600 % 24:.2f} hours, {c // 60 % 60:.2f} minutes, {c % 60:.2f} seconds.")
