import os
import torch
import numpy as np
import cv2
import argparse
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import logging

# ------------------------ Setup Logging ------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


# ------------------------ Helper Functions ------------------------

def extract_frames_from_video(video_path, every_n=5, max_frames=16):
    """Extracts frames from a video at a regular interval."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        if count % every_n == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
        count += 1
    cap.release()
    logging.info(f"Extracted {len(frames)} frames from {video_path}")
    return frames


def preprocess_frames(frames, processor):
    """Preprocess frames for CLIP model."""
    inputs = processor(images=frames, return_tensors="pt", padding=True)
    return inputs


def get_image_embeddings(model, inputs):
    """Generate image embeddings from CLIP model."""
    with torch.no_grad():
        image_embeds = model.get_image_features(**inputs)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds


def get_text_embedding(model, processor, text):
    """Generate text embedding from CLIP model."""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeds = model.get_text_features(**inputs)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds[0]


def compute_similarity(image_embeddings, text_embedding):
    """Compute cosine similarity."""
    sims = torch.matmul(image_embeddings, text_embedding)
    return sims


def visualize_similarities(frames, similarities, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, (frame, sim) in enumerate(zip(frames, similarities)):
        plt.imshow(frame)
        plt.title(f"Similarity: {sim.item():.4f}")
        plt.axis('off')
        output_path = os.path.join(output_dir, f"frame_{i}_sim_{sim.item():.4f}.png")
        plt.savefig(output_path)
        plt.close()
    logging.info(f"Saved similarity visualizations to {output_dir}")


# ------------------------ Main Pipeline ------------------------

def run_faithfulness_on_video(video_path, prompt, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Step 1: Extract frames
    frames = extract_frames_from_video(video_path)

    # Step 2: Preprocess
    inputs = preprocess_frames(frames, processor)
    for k in inputs:
        inputs[k] = inputs[k].to(device)

    # Step 3: Image and Text Embeddings
    image_embeds = get_image_embeddings(model, inputs)
    text_embed = get_text_embedding(model, processor, prompt).to(device)

    # Step 4: Compute similarities
    similarities = compute_similarity(image_embeds, text_embed)

    # Step 5: Visualize
    visualize_similarities(frames, similarities, output_dir)

    logging.info("Faithfulness analysis completed successfully.")


# ------------------------ Entry Point ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLIP-based faithfulness analysis on a video.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to compare with video")
    parser.add_argument("--output_dir", type=str, default="output_frames", help="Directory to save visualizations")

    args = parser.parse_args()
    run_faithfulness_on_video(args.video_path, args.prompt, args.output_dir)
