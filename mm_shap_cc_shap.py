import torch
import numpy as np
from scipy import spatial, stats, special
from sklearn import metrics
import copy, random, math, sys
import matplotlib as plt
import shap

from generation_and_prompting import *
from config import *
random.seed(42)

model_name = sys.argv[2]

def compute_mm_score(text_length, shap_values):
    """ Compute Multimodality Score. (80% textual, 20% visual, possibly: 0% knowledge). """
    image_contrib = np.abs(shap_values.values[0, :text_length, :]).sum()
    text_contrib = np.abs(shap_values.values[0, text_length:, :]).sum()
    text_score = text_contrib / (text_contrib + image_contrib)
    # image_score = image_contrib / (text_contrib + image_contrib) # is just 1 - text_score in the two modalities case
    return text_score


def explain_VLM(prompt, raw_image, model, tokenizer, max_new_tokens=100, p=None):
    """
    This is the equivalent function of explain_lm. It returns shap_values.
    Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens).
    """
    inputs = tokenizer(prompt, raw_image, return_tensors='pt').to("cuda", torch.float16)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, min_new_tokens=1, do_sample=True)
    output_ids = outputs[:, inputs.input_ids.shape[1]:].to('cpu') # select only the output ids without repeating the input again
    inputs.to('cpu')

    def custom_masker(mask, x):
        """
        Shap relevant function.
        It gets a mask from the shap library with truth values about which image and text tokens to mask (False) and which not (True).
        It defines how to mask the text tokens and masks the text tokens. So far, we don't mask the image, but have only defined which image tokens to mask. The image tokens masking happens in get_model_prediction().

        Token significance of the BakLLaVA model and the llava_mistral:
        1: <s> (start of sequence)
        32000: <image> (image token) (28705, 32000, 28705)
        583: _
        Token significance of the llava_vicuna model:
        1: <s> (start of sequence)
        32000: <image> (image token) (29871, 32000, 29871)
        903: _
        """
        masked_X = x.clone() # x.shape is (num_permutations, img_length+text_length)
        # never mask out <s> and <image> tokens (makes no sense for the model to work without them)
        # find all ids of masked_X where the value is 1 or 32000 since we are not going to mask these special tokens
        if model_name == "llava_vicuna":
            condition = (masked_X == 1) | (masked_X == 32000) | (masked_X == 29871)
        else: # bakllava and llava_mistral specific
            condition = (masked_X == 1) | (masked_X == 32000) | (masked_X == 28705)
        indices = torch.nonzero(condition, as_tuple=False)
        mask[indices] = True

        # set to zero the image tokens we are going to mask
        image_mask = torch.tensor(mask).unsqueeze(0)
        image_mask[:, -nb_text_tokens:] = True # do not mask text tokens yet
        masked_X[~image_mask] = 0  # ~mask !!! to zero

        # mask the text tokens (delete them)
        text_mask = torch.tensor(mask).unsqueeze(0)
        text_mask[:, :-nb_text_tokens] = True # do not do anything to image tokens anymore
        if model_name == "llava_vicuna":
            masked_X[~text_mask] = 903
        else: # bakllava and llava_mistral specific
            masked_X[~text_mask] = 583
        return masked_X#.unsqueeze(0)

    def get_model_prediction(x):
        """
        Shap relevant function.
        1. Mask the image pixel according to the specified patches to mask from the custom masker.
        2. Predict the model output for all combinations of masked image and tokens. This is then further passed to the shap libary.
        """
        with torch.no_grad():
            # split up the input_ids and the image_token_ids from x (containing both appended)
            input_ids = torch.tensor(x[:, -inputs.input_ids.shape[1]:]) # text ids
            masked_image_token_ids = torch.tensor(x[:, :-inputs.input_ids.shape[1]])
            # output_ids.shape is (1, output_length); result.shape is (num_permutations, output_length)
            result = np.zeros((input_ids.shape[0], output_ids.shape[1]))
            
            # call the model for each "new image" generated with masked features
            for i in range(input_ids.shape[0]):
                # here the actual masking of the image is happening. The custom masker only specified which patches to mask, but no actual masking has happened
                masked_inputs = copy.deepcopy(inputs) # initialize the thing
                masked_inputs['input_ids'] = input_ids[i].unsqueeze(0)

                if model_name != "bakllava":
                    raw_image_arr = np.array(raw_image)
                    if len(raw_image_arr.shape) == 2:
                        raw_image_arr = np.array(raw_image.convert("RGB"))
                        # raw_image_array = raw_image_array[:, :, np.newaxis] 
                    raw_image_array = copy.deepcopy(raw_image_arr)

                # pathify the image
                for k in range(masked_image_token_ids[i].shape[0]):
                    if masked_image_token_ids[i][k] == 0:  # should be zero
                        m = k // p
                        n = k % p
                        if model_name == "bakllava":
                            masked_inputs["pixel_values"][:, :, m *
                                patch_size:(m+1)*patch_size, n*patch_size:(n+1)*patch_size] = 0 # torch.rand(3, patch_size, patch_size)  # np.random.rand()
                        else:
                            raw_image_array[m*patch_size:(m+1)*patch_size, n*patch_size:(n+1)*patch_size, :] = 0
                if model_name != "bakllava":
                    masked_pixel_vals = tokenizer(prompt, raw_image_array, return_tensors='pt').pixel_values
                    masked_inputs["pixel_values"] = masked_pixel_vals
                
                masked_inputs.to("cuda", torch.float16)
                # # generate outputs and logits
                out = model.generate(**masked_inputs, max_new_tokens=max_new_tokens,do_sample=False, output_logits=True, output_scores=True, return_dict_in_generate=True)
                logits = out.logits[0].detach().cpu().numpy()
                # extract only logits corresponding to target sentence ids
                result[i] = logits[0, output_ids]
        return result

    nb_text_tokens = inputs.input_ids.shape[1] # number of text tokens
    if p is None: # calculate the sqrt(number) of patches needed to cover the image
        p = int(math.ceil(np.sqrt(nb_text_tokens)))
    patch_size = inputs.pixel_values.shape[-1] // p # inputs.pixel_values.shape -> torch.Size([1, 3, 336, 336]) for BakLLaVA
    # give the image patches some token ids and make them negative to distinguish them from text tokens
    image_token_ids = torch.tensor(range(-1, -p**2-1, -1)).unsqueeze(0)

    # make a combination between tokens and pixel_values (transform to patches first)
    X = torch.cat((image_token_ids, inputs.input_ids), 1).unsqueeze(1)
    try:
        explainer = shap.Explainer(get_model_prediction, custom_masker, silent=True, max_evals=600)
        shap_values = explainer(X)[0]
    except ValueError:
        try:
            explainer = shap.Explainer(get_model_prediction, custom_masker, silent=True, max_evals=700)
            shap_values = explainer(X)[0]
        except ValueError:
            try:
                explainer = shap.Explainer(get_model_prediction, custom_masker, silent=True, max_evals=800)
                shap_values = explainer(X)[0]
            except ValueError:
                explainer = shap.Explainer(get_model_prediction, custom_masker, silent=True, max_evals=900)
                shap_values = explainer(X)[0]

    if len(shap_values.values.shape) == 2:
        shap_values.values = np.expand_dims(shap_values.values, axis=2)

    mm_score = compute_mm_score(nb_text_tokens, shap_values)

    return shap_values, mm_score, p, nb_text_tokens

# prompt = "USER: <image>\nWhat is this? (A): a pizza, or (B): a dog. \nASSISTANT: The answer is: ("
# image_path = "/home/mitarb/parcalabescu/COCO/all_images/COCO_test2014_000000489547.jpg"
# raw_image = Image.open(image_path)
# explain_VLM(prompt, raw_image, model, tokenizer)

def aggregate_values_explanation(shap_values, tokenizer, to_marginalize =' Yes. Why?'):
    """ Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens)."""
    # aggregate the values for the first input token
    # want to get 87 values (aggregate over the whole output)
    # ' Yes', '.', ' Why', '?' are not part of the values we are looking at (marginalize into base value using SHAP property)
    len_to_marginalize = tokenizer(to_marginalize).input_ids.shape[1] - 2 # -2 because tokenizer adds the <s> token here again and a space 28705
    add_to_base = np.abs(shap_values.values[:, -len_to_marginalize:]).sum(axis=1)
    # check if values per output token are not very low
    small_values = [True if x < 0.01 else False for x in np.mean(np.abs(shap_values.values[0,-len_to_marginalize:]), axis=0)]
    if any(small_values):
        print("Warning: Some output expl. tokens have very low values. This might be a problem because they will be rendered large by normalization.")
    # convert shap_values to ratios accounting for the different base values and predicted token probabilities between explanations
    ratios = shap_values.values / (np.abs(shap_values.values).sum(axis=1) - add_to_base) * 100
    # take only the input tokens (without the explanation prompting ('Yes. Why?'))
    return np.mean(ratios, axis=2)[0, :-len_to_marginalize] # we only have one explanation example in the batch

def aggregate_values_prediction(shap_values):
    """ Shape of shap_vals tensor (num_sentences, num_input_tokens, num_output_tokens). """
    # model_output = shap_values.base_values + shap_values.values.sum(axis=1)
    ratios = shap_values.values /  np.abs(shap_values.values).sum(axis=1) * 100
    return np.mean(ratios, axis=2)[0] # we only have one explanation example in the batch

def cc_shap_score(ratios_prediction, ratios_explanation):
    cosine = spatial.distance.cosine(ratios_prediction, ratios_explanation)
    distance_correlation = spatial.distance.correlation(ratios_prediction, ratios_explanation)
    mse = metrics.mean_squared_error(ratios_prediction, ratios_explanation)
    var = np.sum(((ratios_prediction - ratios_explanation)**2 - mse)**2) / ratios_prediction.shape[0]
    
    # how many bits does one need to encode P using a code optimised for Q. In other words, encoding the explanation from the answer
    kl_div = stats.entropy(special.softmax(ratios_explanation), special.softmax(ratios_prediction))
    js_div = spatial.distance.jensenshannon(special.softmax(ratios_prediction), special.softmax(ratios_explanation))

    return cosine, distance_correlation, mse, var, kl_div, js_div

def compute_cc_shap(values_prediction, values_explanation, tokenizer, num_patches_x, nb_text_tokens_pred, nb_text_tokens_expl, marg_pred='', marg_expl=' Yes. Why?'):
    if marg_pred == '':
        ratios_prediction = aggregate_values_prediction(values_prediction)
    else:
        ratios_prediction = aggregate_values_explanation(values_prediction, tokenizer, marg_pred)
    ratios_explanation = aggregate_values_explanation(values_explanation, tokenizer, marg_expl)

    cosine, dist_correl, mse, var, kl_div, js_div = cc_shap_score(ratios_prediction, ratios_explanation)

    input_tokens = values_prediction.data[0].tolist()[-nb_text_tokens_pred:]
    text_tokens = [tokenizer.decode([x], skip_special_tokens=False) for x in input_tokens]
    expl_input_tokens = values_explanation.data[0].tolist()[-nb_text_tokens_expl:]
    expl_text_tokens = [tokenizer.decode([x], skip_special_tokens=False) for x in expl_input_tokens]
    shap_plot_info = {
        'input_tokens': text_tokens, # more than we have values for because of len_marg_pred
        'ratios_prediction': ratios_prediction.astype(float).round(2).tolist(),
        'expl_input_tokens': expl_text_tokens,
        'ratios_explanation': ratios_explanation.astype(float).round(2).tolist(),
        'num_patches_x': num_patches_x, # the number of patches in x direction
        # 'nb_text_tokens': nb_text_tokens_pred,
    }
    
    return cosine, dist_correl, mse, var, kl_div, js_div, shap_plot_info

def cc_shap_measure(inp_ask_for_prediction, prediction, input_pred_ask_for_expl, raw_image, model, tokenizer, c_task, tuple_shap_values_prediction=None, expl_type='post_hoc', max_new_tokens=100):
    """ Measure idea: Let the model make a prediction. Let the model explain and compare the input contributions
      for prediction and explanation. CC-SHAP takes a continuous value $\in [-1,1]$, where higher is more self-consistent.
      Returns a high score (1) for self-consistent (faithful) answers and a low score for unfaithful answers (-1). """

    if expl_type == 'post_hoc':
        pred_ask_for_expl = prompt_post_hoc_expl(prediction, c_task)
    elif expl_type == 'cot':
        pred_ask_for_expl = prompt_cot(c_task)
    
    # make sure the explanation uses the same number of image patches as before (for prediction)
    if tuple_shap_values_prediction is None:
        shap_values_prediction, mm_score, num_patches_x, nb_text_tokens_pred = explain_VLM(inp_ask_for_prediction, raw_image, model, tokenizer, max_new_tokens=1) # also compute MM-SHAP here
    else:
        shap_values_prediction, mm_score, num_patches_x, nb_text_tokens_pred = tuple_shap_values_prediction
    shap_values_explanation, mm_score_expl, _ , nb_text_tokens_expl = explain_VLM(input_pred_ask_for_expl, raw_image, model, tokenizer, max_new_tokens=max_new_tokens, p=num_patches_x)

    scores = compute_cc_shap(shap_values_prediction, shap_values_explanation, tokenizer, num_patches_x, nb_text_tokens_pred, nb_text_tokens_expl, marg_pred=prompt_answer(c_task), marg_expl=pred_ask_for_expl)
    
    cosine, distance_correlation, mse, var, kl_div, js_div, shap_plot_info = scores
    return mm_score, mm_score_expl, 1 - cosine, 1 - distance_correlation, 1 - mse, 1 - var, 1 - kl_div, 1 - js_div, shap_plot_info, (shap_values_prediction, mm_score, num_patches_x, nb_text_tokens_pred)

# cc_shap_measure('When do I enjoy walking with my cute dog? On (A): a rainy day, or (B): a sunny day.', model, tokenizer, c_task, labels=['X', 'A', 'B', 'var' ,'C', 'Y'], expl_type='post_hoc')

