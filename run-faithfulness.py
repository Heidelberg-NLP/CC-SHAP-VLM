import time, sys
import torch
print("Cuda is available:", torch.cuda.is_available())
from accelerate import Accelerator
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
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
save_json = int(sys.argv[4])
data_root = sys.argv[5]

model, tokenizer = load_models(model_name)

# prompt = "USER: <image>\nWhat is this?\nASSISTANT:"
# image_path = "/home/mitarb/parcalabescu/COCO/all_images/COCO_test2014_000000489547.jpg"
# raw_image = Image.open(image_path)
# print(vlm_generate(prompt, raw_image, model, tokenizer, max_new_tokens=max_new_tokens))

# prompt = "USER: <image>\nWhat is this? (A): a pizza, or (B): a dog. \nASSISTANT: The answer is: ("
# image_path = "/home/mitarb/parcalabescu/COCO/all_images/COCO_test2014_000000489547.jpg"
# raw_image = Image.open(image_path)
# print(vlm_classify(prompt, raw_image, model, tokenizer, labels=['Y', 'X', 'A', 'B', 'var' ,'Y']))
# print(f"This script so far (generation) needed {time.time()-t1:.2f}s.")

if 'atanasova_counterfactual' in TESTS or 'turpin' in TESTS or 'lanham' in TESTS:
    with torch.no_grad():
        helper_model = AutoModelForCausalLM.from_pretrained(MODELS['llama2-13b-chat'], torch_dtype=torch.float16, device_map="auto", token=True)
    helper_tokenizer = AutoTokenizer.from_pretrained(MODELS['llama2-13b-chat'], use_fast=False, padding_side='left')
    print(f"Loaded helper model {time.time()-t1:.2f}s.")
else:
    print(f"No need for helper model given the subselection of tests.")

# print(lm_generate('I enjoy walking with my cute dog.', helper_model, helper_tokenizer, max_new_tokens=max_new_tokens))


if __name__ == '__main__':
    ############################# run experiments on data
    res_dict = {}
    formatted_samples, correct_answers, wrong_answers, image_paths = [], [], [], []
    accuracy, accuracy_cot = 0, 0
    atanasova_counterfact_count, turpin_test_count, count, cc_shap_post_hoc_sum, cc_shap_cot_sum = 0, 0, 0, 0, 0
    mm_shap_post_hoc_sum, mm_shap_expl_post_hoc_sum, mm_shap_cot_sum, mm_shap_expl_cot_sum = 0, 0, 0, 0
    lanham_early_count, lanham_mistake_count, lanham_paraphrase_count, lanham_filler_count = 0, 0, 0, 0

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

                # shuffle the order of caption and foil such that the correct answer is not always A
                if random.choice([0, 1]) == 0:
                    formatted_sample = format_example_valse_pairwise(test_sentences[0], test_sentences[1])
                    correct_answer, wrong_answer = 'A', 'B'
                else:
                    formatted_sample = format_example_valse_pairwise(test_sentences[1], test_sentences[0])
                    correct_answer, wrong_answer = 'B', 'A'

                formatted_samples.append(formatted_sample)
                correct_answers.append(correct_answer)
                wrong_answers.append(wrong_answer)
                image_paths.append(test_img_path)

                count += 1

    elif c_task in OPEN_ENDED_DATA.keys(): # open ended generation tasks
        images_path = f"{data_root}{OPEN_ENDED_DATA[c_task][0]}"
        qa_path = f"{data_root}{OPEN_ENDED_DATA[c_task][1]}"
        vqa_data = read_data(c_task, qa_path, images_path, data_root)
        for foil_id, foil in tqdm(vqa_data.items()):  # tqdm
            if count + 1 > num_samples:
                break
            test_img_path = os.path.join(images_path, foil["image_file"])
            question = foil["caption"]
            formatted_sample = format_example_vqa_gqa(question) # takes in question
            if c_task == 'vqa':
                correct_answer = foil["answers"] # there are multiple answers annotations
            else:
                correct_answer = foil["answer"]
            wrong_answer = "impossible to give"

            formatted_samples.append(formatted_sample)
            correct_answers.append(correct_answer)
            wrong_answers.append(wrong_answer)
            image_paths.append(test_img_path)

            count += 1
    else:
        raise NotImplementedError(f'Your specified task has no implementation: {c_task}')
    
    print("Done preparing data. Running test...")
    for k, formatted_sample, correct_answer, wrong_answer, image_path in tqdm(zip(range(len(formatted_samples)), formatted_samples, correct_answers, wrong_answers, image_paths)):
        raw_image = Image.open(image_path) # read image
        if c_task in MULT_CHOICE_DATA.keys():
            labels = LABELS['binary']
        elif c_task in OPEN_ENDED_DATA.keys():
            labels = None
        else:
            labels = LABELS[c_task]
        # compute model accuracy post-hoc
        inp_ask_for_prediction = prompt_answer_with_input(formatted_sample, c_task)
        prediction = vlm_predict(inp_ask_for_prediction, raw_image, model, tokenizer, c_task, labels=labels)
        accuracy_sample = evaluate_prediction(prediction, correct_answer, c_task)
        accuracy += accuracy_sample
        # post-hoc explanation
        input_pred_ask_for_expl = prompt_post_hoc_expl_with_input(formatted_sample, prediction, c_task)
        input_pred_expl = vlm_generate(input_pred_ask_for_expl, raw_image, model, tokenizer, max_new_tokens=max_new_tokens, repeat_input=True)

        # for accuracy with CoT: first let the model generate the cot, then the answer.
        input_ask_for_cot = prompt_cot_with_input(formatted_sample, c_task)
        input_cot = vlm_generate(input_ask_for_cot, raw_image, model, tokenizer, max_new_tokens=max_new_tokens, repeat_input=True)
        input_cot_ask_for_pred = prompt_answer_after_cot_with_input(input_cot, c_task)
        prediction_cot = vlm_predict(input_cot_ask_for_pred, raw_image, model, tokenizer, c_task, labels=labels)
        accuracy_cot_sample = evaluate_prediction(prediction_cot, correct_answer, c_task)
        accuracy_cot += accuracy_cot_sample

        # # post-hoc tests
        if 'atanasova_counterfactual' in TESTS:
            atanasova_counterfact, atanasova_counterfact_info = faithfulness_test_atanasova_etal_counterfact(formatted_sample, raw_image, prediction, model, tokenizer, c_task, helper_model, helper_tokenizer, labels)
        else: atanasova_counterfact, atanasova_counterfact_info = 0, 0
        if 'cc_shap-posthoc' in TESTS:
            mm_score_post_hoc, mm_score_expl_post_hoc, score_post_hoc, dist_correl_ph, mse_ph, var_ph, kl_div_ph, js_div_ph, shap_plot_info_ph, tuple_shap_values_prediction = cc_shap_measure(inp_ask_for_prediction, prediction, input_pred_ask_for_expl, raw_image, model, tokenizer, c_task, tuple_shap_values_prediction=None, expl_type='post_hoc', max_new_tokens=max_new_tokens)
        else: mm_score_post_hoc, mm_score_expl_post_hoc, score_post_hoc, dist_correl_ph, mse_ph, var_ph, kl_div_ph, js_div_ph, shap_plot_info_ph = 0, 0, 0, 0, 0, 0, 0, 0, 0

        # # CoT tests
        if 'turpin' in TESTS:
            turpin, turpin_info = faithfulness_test_turpin_etal(formatted_sample, input_cot_ask_for_pred, prediction_cot, raw_image, prediction_cot, correct_answer, wrong_answer, model, tokenizer, c_task, helper_model, helper_tokenizer, labels, max_new_tokens=max_new_tokens)
        else: turpin, turpin_info = 0, 0
        if 'lanham' in TESTS:
            lanham_early, lanham_mistake, lanham_paraphrase, lanham_filler, lanham_early_info = faithfulness_test_lanham_etal(prediction_cot, input_cot, input_ask_for_cot, raw_image, model, tokenizer, c_task, helper_model, helper_tokenizer, labels, max_new_tokens=max_new_tokens)
        else: lanham_early, lanham_mistake, lanham_paraphrase, lanham_filler, lanham_early_info = 0, 0, 0, 0, 0
        if 'cc_shap-cot' in TESTS:
            mm_score_cot, mm_score_expl_cot, score_cot, dist_correl_cot, mse_cot, var_cot, kl_div_cot, js_div_cot, shap_plot_info_cot, _ = cc_shap_measure(inp_ask_for_prediction, prediction, input_ask_for_cot, raw_image, model, tokenizer, c_task, tuple_shap_values_prediction, expl_type='cot', max_new_tokens=max_new_tokens)
        else: mm_score_cot, mm_score_expl_cot, score_cot, dist_correl_cot, mse_cot, var_cot, kl_div_cot, js_div_cot, shap_plot_info_cot = 0, 0, 0, 0, 0, 0, 0, 0, 0

        # aggregate results
        atanasova_counterfact_count += atanasova_counterfact
        cc_shap_post_hoc_sum += score_post_hoc
        turpin_test_count += turpin
        lanham_early_count += lanham_early
        lanham_mistake_count += lanham_mistake
        lanham_paraphrase_count += lanham_paraphrase
        lanham_filler_count += lanham_filler
        cc_shap_cot_sum += score_cot
        mm_shap_post_hoc_sum += mm_score_post_hoc
        mm_shap_expl_post_hoc_sum += mm_score_expl_post_hoc
        mm_shap_cot_sum += mm_score_cot
        mm_shap_expl_cot_sum += mm_score_expl_cot

        res_dict[f"{c_task}_{model_name}_{k}"] = {
            "image_path": image_path,
            "sample": formatted_sample,
            "correct_answer": correct_answer,
            "post-hoc": {
                "inp_pred_expl": input_pred_expl, # input, prediction, expl
                "prediction": prediction,
                "accuracy": accuracy_sample,
                "shap_plot_info_post_hoc": shap_plot_info_ph,
                "cc_shap-posthoc": f"{score_post_hoc:.2f}",
                "t-shap_post_hoc": f"{mm_score_post_hoc*100:.0f}",
                "t-shap_expl_post_hoc": f"{mm_score_expl_post_hoc*100:.0f}",
                "atanasova_counterfact": atanasova_counterfact,
                "atanasova_counterfact_info": atanasova_counterfact_info,
                "other_measures_post_hoc": {
                    "dist_correl": f"{dist_correl_ph:.2f}",
                    "mse": f"{mse_ph:.2f}",
                    "var": f"{var_ph:.2f}",
                    "kl_div": f"{kl_div_ph:.2f}",
                    "js_div": f"{js_div_ph:.2f}"
                },
            },
            "cot": {
                "inp_cot_askpred": input_cot_ask_for_pred, # input, generated cot, prompt for final answer
                "pred_cot": prediction_cot, # prediction after cot
                "accuracy_cot": accuracy_cot_sample,
                "shap_plot_info_cot": shap_plot_info_cot,
                # add plot info for the rest of the tests as well
                "turpin": turpin,
                "turpin_info": turpin_info,
                "lanham_early": lanham_early,
                "lanham_early_info": lanham_early_info,
                "lanham_mistake": lanham_mistake,
                "lanham_paraphrase": lanham_paraphrase,
                "lanham_filler": lanham_filler,
                "cc_shap-cot": f"{score_cot:.2f}",
                "t-shap_cot": f"{mm_score_cot*100:.0f}",
                "t-shap_expl_cot": f"{mm_score_expl_cot*100:.0f}",
                "other_measures_cot": {
                    "dist_correl": f"{dist_correl_cot:.2f}",
                    "mse": f"{mse_cot:.2f}",
                    "var": f"{var_cot:.2f}",
                    "kl_div": f"{kl_div_cot:.2f}",
                    "js_div": f"{js_div_cot:.2f}"
                },
            }
        }

        # write results every 10 samples
        if (k+1) % 10 == 0:
            print(f"Ran {TESTS} on {c_task} {k+1} samples with model {model_name}. Reporting accuracy and faithfulness percentage.\n")
            print(f"Accuracy %                       : {accuracy*100/(k+1):.2f}  ")
            print(f"Atanasova Counterfact %          : {atanasova_counterfact_count*100/(k+1):.2f}  ")
            print(f"CC-SHAP post-hoc mean score      : {cc_shap_post_hoc_sum/(k+1):.2f}  ")
            print(f"Accuracy CoT %                   : {accuracy_cot*100/(k+1):.2f}  ")
            print(f"Turpin %                         : {turpin_test_count*100/(k+1):.2f}  ")
            print(f"Lanham Early Answering %         : {lanham_early_count*100/(k+1):.2f}  ")
            print(f"Lanham Filler %                  : {lanham_filler_count*100/(k+1):.2f}  ")
            print(f"Lanham Mistake %                 : {lanham_mistake_count*100/(k+1):.2f}  ")
            print(f"Lanham Paraphrase %              : {lanham_paraphrase_count*100/(k+1):.2f}  ")
            print(f"CC-SHAP CoT mean score           : {cc_shap_cot_sum/(k+1):.2f}  ")
            print(f"T-SHAP post-hoc mean score %     : {mm_shap_post_hoc_sum/(k+1)*100:.2f}  ")
            print(f"T-SHAP expl post-hoc mean score %: {mm_shap_expl_post_hoc_sum/(k+1)*100:.2f}  ")
            print(f"T-SHAP CoT mean score %          : {mm_shap_cot_sum/(k+1)*100:.2f}  ")
            print(f"T-SHAP expl CoT mean score %     : {mm_shap_expl_cot_sum/(k+1)*100:.2f}  ")
        if save_json and (k+1) % 10 == 0:
            # save results to a json file, make results_json directory if it does not exist
            if not os.path.exists('results_json'):
                os.makedirs('results_json')
            with open(f"results_json/{c_task}_{model_name}_{k+1}.json", 'w') as file:
                json.dump(res_dict, file)

    if save_json:
        # save results to a json file, make results_json directory if it does not exist
        if not os.path.exists('results_json'):
            os.makedirs('results_json')
        with open(f"results_json/{c_task}_{model_name}_{count}_final.json", 'w') as file:
            json.dump(res_dict, file)


    print(f"Ran {TESTS} on {c_task} {count} samples with model {model_name}. Reporting accuracy and faithfulness percentage.\n")
    print(f"Accuracy %                       : {accuracy*100/count:.2f}  ")
    print(f"Atanasova Counterfact %          : {atanasova_counterfact_count*100/count:.2f}  ")
    print(f"CC-SHAP post-hoc mean score      : {cc_shap_post_hoc_sum/count:.2f}  ")
    print(f"Accuracy CoT %                   : {accuracy_cot*100/count:.2f}  ")
    print(f"Turpin %                         : {turpin_test_count*100/count:.2f}  ")
    print(f"Lanham Early Answering %         : {lanham_early_count*100/count:.2f}  ")
    print(f"Lanham Filler %                  : {lanham_filler_count*100/count:.2f}  ")
    print(f"Lanham Mistake %                 : {lanham_mistake_count*100/count:.2f}  ")
    print(f"Lanham Paraphrase %              : {lanham_paraphrase_count*100/count:.2f}  ")
    print(f"CC-SHAP CoT mean score           : {cc_shap_cot_sum/count:.2f}  ")
    print(f"T-SHAP post-hoc mean score %     : {mm_shap_post_hoc_sum/count*100:.2f}  ")
    print(f"T-SHAP expl post-hoc mean score %: {mm_shap_expl_post_hoc_sum/count*100:.2f}  ")
    print(f"T-SHAP CoT mean score %          : {mm_shap_cot_sum/count*100:.2f}  ")
    print(f"T-SHAP expl CoT mean score %     : {mm_shap_expl_cot_sum/count*100:.2f}  ")


    c = time.time()-t1
    print(f"\nThis script ran for {c // 86400:.2f} days, {c // 3600 % 24:.2f} hours, {c // 60 % 60:.2f} minutes, {c % 60:.2f} seconds.")
