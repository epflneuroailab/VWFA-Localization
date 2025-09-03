"""
Extract activations from Qwen2-VL model for VWFA localizer stimuli.

This script loads a pretrained Qwen2-VL model, registers hooks to capture
layer activations, and saves averaged activations for words vs. non-words.
Reproducibility is ensured by fixing seeds and using deterministic settings.
"""

# --- Imports ---
import os
import random
import base64
import pickle as pkl
from glob import glob
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from accelerate import Accelerator
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==============================
# 1. Reproducibility Settings
# ==============================
BASE_SEED = 42

random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
torch.cuda.manual_seed(BASE_SEED)
torch.cuda.manual_seed_all(BASE_SEED)

os.environ["PYTHONHASHSEED"] = str(BASE_SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==============================
# 2. Paths & Constants
# ==============================
SAVE_PATH = "./saved_model"
SAVE_DIR = "./activations"
LI_DIR = "./VWFA_localizer_contrast_classes"

os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 1
NUM_SAMPLES = 288  # Number of images per iteration
NUM_ITERATIONS = 5  # Number of repeated runs
NUM_PER_CATEGORY = NUM_SAMPLES // 4

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==============================
# 3. Model & Processor
# ==============================
accelerator = Accelerator()

model = Qwen2VLForConditionalGeneration.from_pretrained(
    SAVE_PATH,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    offload_folder="./offload",
)
model.gradient_checkpointing_enable()
processor = AutoProcessor.from_pretrained(SAVE_PATH)

model = accelerator.prepare(model)
model.eval()

print("Model loaded, distributed, and ready!")

# ==============================
# 4. Dataset
# ==============================
class JLiVWFADataset(Dataset):
    """VWFA Localizer dataset: words, faces, lines, scrambled words."""

    def __init__(self):
        dirpath = os.path.expanduser(LI_DIR)

        word_paths = glob(f"{dirpath}/VWFA_class_1_word_pngs_grid/*.png")
        face_paths = glob(f"{dirpath}/VWFA_class_2_face_pngs_grid/*.png")
        line_paths = glob(f"{dirpath}/VWFA_class_2_line_pngs_grid/*.png")
        scr_paths = glob(f"{dirpath}/VWFA_class_2_scr_word_pngs_grid/*.png")

        print(f"#Words: {len(word_paths)} | #Faces: {len(face_paths)} | "
              f"#Lines: {len(line_paths)} | #Scrambled: {len(scr_paths)}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.items = (
            [(p, "W") for p in word_paths] +
            [(p, "F") for p in face_paths] +
            [(p, "L") for p in line_paths] +
            [(p, "S") for p in scr_paths]
        )
        self.items = sorted(self.items, key=lambda x: x[1])

        print(f"Total images: {len(self.items)}")

    def __getitem__(self, idx):
        img_path, label = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        base64_image = encode_image_base64(img_path)
        return img_path, label, base64_image

    def __len__(self):
        return len(self.items)


def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ==============================
# 5. Hook Utilities
# ==============================
def _get_layer(model, layer_name):
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer
    raise ValueError(f"Layer '{layer_name}' not found in model.")


def _register_hook(layer, key, target_dict):
    def hook_function(_layer, _input, output, key=key):
        output = output[0] if isinstance(output, (tuple, list)) else output
        target_dict[key] = output
    return layer.register_forward_hook(hook_function)


def setup_hooks(model, layer_names):
    hooks, representations = [], OrderedDict()
    for layer_name in layer_names:
        layer = _get_layer(model, layer_name)
        hook = _register_hook(layer, layer_name, representations)
        hooks.append(hook)
    return hooks, representations


def define_layers(model):
    """Select model layers of interest by suffix."""
    target_endings = ["mlp.gate_proj"]
    return [
        name for name, _ in model.named_modules()
        if any(name.endswith(e) for e in target_endings)
    ]


# ==============================
# 6. Activation Extraction
# ==============================
dataset = JLiVWFADataset()
layer_names = define_layers(model)

# Split dataset into categories
words_indices = [i for i, (_, l) in enumerate(dataset.items) if l == "W"]
faces_indices = [i for i, (_, l) in enumerate(dataset.items) if l == "F"]
lines_indices = [i for i, (_, l) in enumerate(dataset.items) if l == "L"]
scr_words_indices = [i for i, (_, l) in enumerate(dataset.items) if l == "S"]

with torch.no_grad():
    for run_idx in range(NUM_ITERATIONS):
        # Set run-specific seeds
        seed = BASE_SEED + run_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        activations = defaultdict(lambda: {"words": [], "non-words": []})
        final_reps = {
            "words": {ln: [] for ln in layer_names},
            "non-words": {ln: [] for ln in layer_names},
        }

        hooks, layer_reps = setup_hooks(model, layer_names)

        # Sample balanced subset
        sampled_indices = (
            random.sample(words_indices, NUM_PER_CATEGORY) +
            random.sample(faces_indices, NUM_PER_CATEGORY) +
            random.sample(lines_indices, NUM_PER_CATEGORY) +
            random.sample(scr_words_indices, NUM_PER_CATEGORY)
        )
        random.shuffle(sampled_indices)

        dataloader = DataLoader(
            Subset(dataset, sampled_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=32,
        )

        # Forward pass
        for image_path, label, base64_image in tqdm(dataloader):
            image = Image.open(image_path[0])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe the image."},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            _ = model(**inputs, output_hidden_states=True)

            input_type = "words" if label[0] == "W" else "non-words"

            for layer_name, tensor in layer_reps.items():
                activation = tensor[0]
                if activation.ndim > 1:
                    activation = activation.mean(axis=0)
                final_reps[input_type][layer_name].append(activation)

            # clear for next batch
            for k in layer_reps:
                layer_reps[k] = None

        # Convert to numpy
        for layer_name in layer_names:
            for input_type in ["words", "non-words"]:
                final_reps[input_type][layer_name] = np.array([
                    act.cpu().to(dtype=torch.float32).numpy()
                    for act in final_reps[input_type][layer_name]
                ])

        # Save to disk
        savepath = os.path.join(SAVE_DIR, f"qwen_activation_{run_idx}.pkl")
        with open(savepath, "wb") as f:
            pkl.dump(final_reps, f)

        print(f" Run {run_idx}: Activations saved to {savepath}")

        for hook in hooks:
            hook.remove()
