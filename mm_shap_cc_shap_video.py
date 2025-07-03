import torch
import numpy as np
import copy

def compute_mmshap_video(model, video_tensor, text_input, mask_fn_video, mask_fn_text, num_samples=20):
    """
    MM-SHAP for video-language models

    Args:
        model: video-text model that outputs logits or scores
        video_tensor: torch.Tensor of shape [T, C, H, W]
        text_input: tokenized text
        mask_fn_video: function to apply spatiotemporal video masks
        mask_fn_text: function to apply text token masks
        num_samples: number of perturbation samples

    Returns:
        shap_values_video: importance per frame or patch
        shap_values_text: importance per token
    """
    base_output = model(video_tensor.unsqueeze(0), text_input)  # shape: [1, num_classes] or similar

    video_contributions = []
    text_contributions = []

    for _ in range(num_samples):
        perturbed_video = mask_fn_video(video_tensor)
        perturbed_text = mask_fn_text(text_input)

        output = model(perturbed_video.unsqueeze(0), perturbed_text)

        delta = output - base_output

        video_score = score_video_mask_effect(video_tensor, perturbed_video)
        text_score = score_text_mask_effect(text_input, perturbed_text)

        video_contributions.append(video_score * delta.item())
        text_contributions.append(text_score * delta.item())

    shap_values_video = np.mean(video_contributions, axis=0)
    shap_values_text = np.mean(text_contributions, axis=0)

    return shap_values_video, shap_values_text

def mask_fn_video(video_tensor, mode='frame', mask_ratio=0.3):
    """
    Randomly masks parts of the video tensor.
    Args:
        video_tensor: torch.Tensor [T, C, H, W]
        mode: 'frame' or 'tube'
    """
    video = video_tensor.clone()
    T, C, H, W = video.shape
    num_to_mask = int(mask_ratio * T)

    if mode == 'frame':
        masked_idxs = np.random.choice(T, num_to_mask, replace=False)
        for t in masked_idxs:
            video[t] = 0
    elif mode == 'tube':
        x_start = np.random.randint(0, W // 2)
        y_start = np.random.randint(0, H // 2)
        patch_size = W // 4
        for t in range(T):
            video[t, :, y_start:y_start + patch_size, x_start:x_start + patch_size] = 0

    return video

def mask_fn_text(text_input, mask_ratio=0.3):
    """
    Randomly masks a portion of the text input tokens.
    text_input: tensor or list of tokens
    """
    masked = copy.deepcopy(text_input)
    num_tokens = len(masked)
    num_to_mask = int(mask_ratio * num_tokens)
    masked_idxs = np.random.choice(num_tokens, num_to_mask, replace=False)

    for idx in masked_idxs:
        masked[idx] = 0  # assuming 0 is the mask token ID
    return masked

def score_video_mask_effect(original, perturbed):
    """Simple heuristic: proportion of masked frames."""
    T = original.shape[0]
    num_masked = torch.sum((original != perturbed).any(dim=1).any(dim=1).any(dim=1)).item()
    return num_masked / T

def score_text_mask_effect(original, perturbed):
    """Simple heuristic: proportion of masked tokens."""
    total_tokens = len(original)
    num_masked = sum(1 for o, p in zip(original, perturbed) if o != p)
    return num_masked / total_tokens

def plot_temporal_shap(shap_values_video):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 2))
    plt.plot(shap_values_video)
    plt.title("Frame-wise SHAP Values")
    plt.xlabel("Frame Index")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.show()
