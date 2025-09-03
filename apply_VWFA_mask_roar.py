"""
This script evaluates the effect of ablating language-sensitive units in Qwen2VL.

1. Loads a precomputed binary mask of top-k language-sensitive units.
2. Applies the mask via forward hooks to the model's mlp.gate_proj layers.
3. Processes a set of images from the ROAR test set.
4. Collects model responses and calculates accuracy against ground truth.
"""

import argparse
import os
import random as rand
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
import pickle as pkl
import scipy.stats as st
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# ------------------ Reproducibility ------------------ #
base_seed = 42
rand.seed(base_seed)
np.random.seed(base_seed)
torch.manual_seed(base_seed)
torch.cuda.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)
os.environ["PYTHONHASHSEED"] = str(base_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
st.seed = base_seed  # Note: SciPy doesn't have a true seed, but we leave it for clarity

sns.set_theme()
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=sns.color_palette("deep"))

# ------------------ Command-line arguments ------------------ #
parser = argparse.ArgumentParser(description="Evaluate Qwen2VL with masked activations")
parser.add_argument('--idx', type=int, default=0, help="Index of the activation file / mask")
parser.add_argument('--percentage', type=int, default=21, help="Percentage of top units to mask")
args = parser.parse_args()
idx, percentage = args.idx, args.percentage

# ------------------ Paths ------------------ #
ZO = True
sample_number = 100
predefined_factor = 0

MASK_PATH = f"./masks/mask_{percentage}%_{idx}.pkl"
RESULTS_PATH = "./muted/Zero_out/roar-test/" if ZO else "./muted/roar-test/"
os.makedirs(RESULTS_PATH, exist_ok=True)
file_path = os.path.join(RESULTS_PATH, f"RoarResults_noised_factor{predefined_factor}_{percentage}%.txt")

# ------------------ Load Model ------------------ #
accelerator = Accelerator()
save_path = "./saved_model"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    save_path, torch_dtype=torch.bfloat16, device_map="balanced", offload_folder="./offload"
)
model.gradient_checkpointing_enable()
processor = AutoProcessor.from_pretrained(save_path)
model = accelerator.prepare(model)
print("Model loaded, distributed, and optimized!")

# ------------------ Load Mask ------------------ #
with open(MASK_PATH, 'rb') as f:
    mask = torch.from_numpy(pkl.load(f)).to(next(model.parameters()).device)

# ------------------ Define Layers ------------------ #
def define_layers(model):
    target_layers = []
    target_endings = ["mlp.gate_proj"]  # Only gate_proj layers
    for name, _ in model.named_modules():
        if any(name.endswith(ending) for ending in target_endings):
            target_layers.append(name)
    return target_layers

# ------------------ Load Ground Truth ------------------ #
def extract_word_type(response_text):
    response_text = response_text.lower()
    if any(s in response_text for s in ["is a pseudo", "is not a real", "appears to be a pseudo word"]):
        return "pseudo"
    if any(s in response_text for s in ["is a real", "is real", "real"]):
        return "real"
    if "pseudo" in response_text:
        return "pseudo"
    return None

def load_ground_truth(csv_file):
    ground_truth = {}
    with open(csv_file, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["word"].strip().lower()
            label = row["realpseudo"].strip().lower()
            ground_truth[word] = label
    return ground_truth

def calculate_accuracy(lines):
    correct_predictions = total_predictions = wrong_real = wrong_pseudo = 0
    for line in lines:
        if not line.strip() or len(line.split(",", 1)) < 2:
            continue
        image_name, response = line.split(",", 1)
        parts = image_name.split("-")
        if len(parts) < 2:
            print(image_name, "wrong")
            continue
        word = parts[1].split(".")[0].strip().lower()
        predicted_label = extract_word_type(response)
        if word in ground_truth:
            total_predictions += 1
            if predicted_label == ground_truth[word]:
                correct_predictions += 1
            else:
                if ground_truth[word] == 'real': wrong_real += 1
                if ground_truth[word] == 'pseudo': wrong_pseudo += 1
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, correct_predictions, total_predictions, wrong_real, wrong_pseudo

# ------------------ Apply Noise Mask ------------------ #
def apply_noise_with_mask(model, mask):
    layer_names = define_layers(model)
    hooks = []
    index = -1
    for name, module in model.named_modules():
        if name in layer_names:
            index += 1
            if index == len(mask):
                break
            def hook_fn(module, input, output, mask=mask[index]):
                output_clone = output
                if isinstance(output, tuple): output = output[0]
                device = output.device
                mask = mask.to(device)
                mask = mask[:output.shape[1]] if output.ndim in [2, 3] else mask
                if output.ndim == 3:
                    adjusted_mask = mask.unsqueeze(0).unsqueeze(2)
                elif output.ndim == 2:
                    adjusted_mask = mask.unsqueeze(0)
                elif output.ndim == 4:
                    adjusted_mask = mask.view(1, -1, 1, 1)
                else:
                    raise ValueError(f"Unsupported output dimensions: {output.shape}")
                scale_factor = 0 if ZO else (predefined_factor or np.random.uniform(0, 1))
                inv_mask = output * (1 - adjusted_mask)
                scaled_output = inv_mask + output * adjusted_mask * scale_factor
                if isinstance(output_clone, tuple):
                    return (scaled_output, *output_clone[1:])
                return scaled_output
            hooks.append(module.register_forward_hook(hook_fn))
    return hooks

hooks = apply_noise_with_mask(model, mask)

# ------------------ Process Images ------------------ #
image_dir = "/mnt/honarman/dyslexia-project-main/stimuli/roar-test-80-20"
image_filenames_all = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
ground_truth_file = "/mnt/honarman/dyslexia-project-main/stimuli/roar-stimuli-1/roar-500-word-list.csv"
ground_truth = load_ground_truth(ground_truth_file)

accuracies = []
lines = []
image_filenames = rand.sample(image_filenames_all, sample_number)
for image_file in tqdm(image_filenames, desc="Processing Images"):
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "A real or pseudo word will be presented to you in an image..."}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    line = f"{image_file}, {output_text[0]}\n"
    print(line)
    lines.append(line)
accuracy, correct, total, wrong_r, wrong_p = calculate_accuracy(lines)
print(f"{accuracy}, ({correct}/{total})\n")
accuracies.append(accuracy)

acc = np.array(accuracies)
mean = np.mean(acc)
std = np.std(acc, ddof=1)
sem = std / np.sqrt(len(acc))
confidence_interval = st.t.interval(0.95, len(acc)-1, loc=mean, scale=sem)
print(f"Mean Accuracy: {mean:.4f}")

with open(file_path, "a") as file:
    file.write(f"Accuracies of mask number{idx}: {acc}, wrong_real: {wrong_r}, wrong_pseudo:{wrong_p}\n")

# Cleanup hooks
for hook in hooks:
    hook.remove()
