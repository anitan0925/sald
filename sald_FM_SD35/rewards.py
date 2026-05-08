import torch
from transformers import CLIPModel, CLIPProcessor, AutoProcessor, Gemma3nForConditionalGeneration
from torch import nn
from huggingface_hub import hf_hub_download
from PIL import Image
from typing import List, Optional, Union, Dict
import numpy as np
# from paddleocr import PaddleOCR
from Levenshtein import distance
import re
import functools
import inspect
import httpx
import re
import io
import concurrent.futures as futures
import time
import httpcore
from google import genai
from google.genai import types
from google.genai.errors import ServerError

import pdb

def retry(times, failed_return, exceptions, backoff_factor=1):
    """A decorator for retrying a function upon specific exceptions."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    # Pass the current attempt number to the decorated function
                    return func(*args, **kwargs, retry_attempt=attempt)
                except exceptions as e:
                    print(
                        f"Exception [{type(e)}:{e}] thrown when attempting to run {func}, attempt {attempt} of {times}"
                    )
                    time.sleep(backoff_factor * 2**attempt)
                    attempt += 1
            return failed_return
        return wrapper
    return decorator


class BaseReward(nn.Module):
    def __init__(self):
        self.num_rewards = 1
        super().__init__()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def __call__(self, images: List[Image.Image], prompts: Optional[List[str]] = None) -> torch.Tensor:
        raise NotImplementedError()

class AestheticMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)
class Aesthetic(BaseReward):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = AestheticMlp()
        cached_path = hf_hub_download("trl-lib/ddpo-aesthetic-predictor", "aesthetic-model.pth")
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"), weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.eval()

    @torch.no_grad()
    def __call__(self, images: List[Image.Image], prompts = None) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

class PickScore(BaseReward):
    def __init__(self):
        super().__init__()
        processor_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_path = "yuvalkirstain/PickScore_v1"
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path)
        self.eval()

    @torch.no_grad()
    def __call__(self, images: List[Image.Image], prompts: List[str]) -> torch.Tensor:
        # Preprocess images
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        # Preprocess text
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        
        # Get embeddings
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        
        # Calculate scores
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        # norm to 0-1
        scores = scores/26
        return scores






############################################################################
from transformers import AutoImageProcessor,CLIPProcessor, CLIPModel
import torchvision.transforms as T
def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")

def get_image_transform(processor:AutoImageProcessor):
    config = processor.to_dict()
    resize = T.Resize(get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
    crop = T.CenterCrop(get_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
    normalise = T.Normalize(mean=processor.image_mean, std=processor.image_std) if config.get("do_normalize") else nn.Identity()

    return T.Compose([resize, crop, normalise])

class CLIPscore(BaseReward):

    def __init__(self):
        super().__init__()
        #self.device=device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14") #.to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tform = get_image_transform(self.processor.image_processor)
        self.eval()
    
    def _process(self, pixels):
        dtype = pixels.dtype
        pixels = self.tform(pixels)
        pixels = pixels.to(dtype=dtype)

        return pixels
    
    def _processImage(self, images):

        if not isinstance(images, torch.Tensor):
            ref_images = [np.array(img) for img in images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        return ref_images


    @torch.no_grad()
    def __call__(self, Images, prompts, return_img_embedding=False):
        texts = self.processor(text=prompts, padding='max_length', truncation=True, return_tensors="pt").to(self.device)
        pixels = self._processImage(Images)
        pixels = self._process(pixels).to(self.device)
        outputs = self.model(pixel_values=pixels, **texts)
        if return_img_embedding:
            return outputs.logits_per_image.diagonal()/30, outputs.image_embeds
        return outputs.logits_per_image.diagonal()/30





##############################################################################


class DiversityScore(BaseReward):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.eval()
    
    
    @torch.no_grad()
    def __call__(self, images: List[Image.Image], prompts = None) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        Pdist = torch.cdist(embed,embed)
        N = embed.shape[0]
        PMD = torch.mean(Pdist) *(N/(N-1))
        # pdb.set_trace()
        return PMD




##############################################################################





# Normalization ranges for each reward type: (min, max)
# After normalization: normalized = (value - min) / (max - min)
REWARD_NORM_RANGES = {
    "aesthetic": (1.0, 10.0),       # MLP output range
    "pickscore": (0.0, 1.0),        # already normalized by /26
}

def normalize_rewards(rewards: torch.Tensor, reward_type: str) -> torch.Tensor:
    """Normalize rewards to [0, 1] based on predefined ranges."""
    if reward_type not in REWARD_NORM_RANGES:
        raise ValueError(f"Unknown reward type: {reward_type}")
    lo, hi = REWARD_NORM_RANGES[reward_type]
    normalized = (rewards - lo) / (hi - lo)
    return normalized

def denormalize_rewards(normalized_rewards: torch.Tensor, reward_type: str) -> torch.Tensor:
    """Convert normalized rewards back to original scale."""
    if reward_type not in REWARD_NORM_RANGES:
        raise ValueError(f"Unknown reward type: {reward_type}")
    lo, hi = REWARD_NORM_RANGES[reward_type]
    rewards = normalized_rewards * (hi - lo) + lo
    return rewards



REWARDS_CLS = {
    "aesthetic": Aesthetic,
    "pickscore": PickScore,
    "CLIPscore": CLIPscore,
    "DiversityScore": DiversityScore,
}