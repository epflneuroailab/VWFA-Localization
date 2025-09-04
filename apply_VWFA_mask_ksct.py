"""
Evaluates Qwen2VL on the KSCT dataset with masked activations.

1. Loads precomputed binary masks for VWF-sensitive units.
2. Applies noise/zero-out to masked units during forward passes.
3. Runs inference on KSCT stimuli with text-image prompts.
4. Computes mean accuracy and confidence intervals.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
import random as rand
import scipy.stats as st
import csv

# ------------------ Reproducibility ------------------
base_seed = 42
rand.seed(base_seed)
np.random.seed(base_seed)
torch.manual_seed(base_seed)
torch.cuda.manual_seed(base_seed)
torch.cuda.manual_seed_all(base_seed)
os.environ["PYTHONHASHSEED"] = str(base_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
st.seed = base_seed 

# ------------------ CLI Args ------------------
parser = argparse.ArgumentParser(description="KSCT masked noise evaluation")
parser.add_argument('--idx', type=int, default=0, help="Mask index")
parser.add_argument('--percentage', type=int, default=10, help="Percentage of units masked")
args = parser.parse_args()

idx = args.idx
percentage = args.percentage

# ------------------ Config ------------------
vision = False
ZO = True
sample_number = 80
predefined_factor = 0

MASK_PATH = "./masks/reproduce_vision_lang_all_tokens"
RESULTS_PATH = "./muted/reproduce_vision_lang_all_tokens"

if not vision:
    MASK_PATH = os.path.join(MASK_PATH, 'NoVision')
    RESULTS_PATH = os.path.join(RESULTS_PATH, 'NoVision')

if ZO:
    RESULTS_PATH = os.path.join(RESULTS_PATH, 'Zero_out')

MASK_PATH = os.path.join(MASK_PATH, f'mask_{percentage}%_{idx}.pkl')
os.makedirs(RESULTS_PATH, exist_ok=True)

file_path = os.path.join(
    RESULTS_PATH,
    f"KSCT_v1_noCap_Results_noised_{percentage}%.txt"
)

# ------------------ Model ------------------
accelerator = Accelerator()
save_path = "./saved_model"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    save_path,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
    offload_folder="./offload"
)
model.gradient_checkpointing_enable()
processor = AutoProcessor.from_pretrained(save_path)
model = accelerator.prepare(model)
print("Model loaded, distributed, and optimized!")

# ------------------ Mask ------------------
def load_language_mask(mask_path):
    with open(mask_path, 'rb') as f:
        return pkl.load(f)

mask = load_language_mask(MASK_PATH)
mask = torch.from_numpy(mask).to(next(model.parameters()).device)

# ------------------ Layer Selection ------------------
def define_layers(model):
    target_endings = ["mlp.gate_proj"]
    target_layers = [
        name for name, _ in model.named_modules()
        if any(name.endswith(ending) for ending in target_endings)
    ]
    if not vision:
        target_layers = [name for name in target_layers if 'visual' not in name]
    return target_layers

# ------------------ Utilities ------------------
def extract_word_type(response_text):
    response_text = response_text.lower()
    a_phrases = [
        "matches photo a", "match photo a", "matches a", "a matches", "**a**",
        "answer is a", "answer is: a", "answer is photo a", "answer is picture a"
    ]
    b_phrases = [
        "matches photo b", "match photo b", "matches b", "b matches", "**b**",
        "answer is b", "answer is: b", "answer is photo b", "answer is picture b"
    ]
    if any(phrase in response_text for phrase in a_phrases):
        return "a"
    if any(phrase in response_text for phrase in b_phrases):
        return "b"
    if " a " in response_text:
        return "a"
    if " b " in response_text:
        return "b"
    return None

def load_ground_truth(csv_file):
    ground_truth = {}
    with open(csv_file, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["\ufeffOrder"].strip().lower()
            label = row["Key"].strip().lower()
            ground_truth[word] = label
    return ground_truth

def load_prompts(csv_file):
    prompts = {}
    with open(csv_file, mode="r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["\ufeffOrder"].strip().lower()
            prompts[word] = row["Description"]
    return prompts

def calculate_accuracy(lines):
    correct_predictions = total_predictions = wrong_a = wrong_b = 0
    for line in lines:
        if not line.strip():
            continue
        if len(line.split(",", 1)) < 2:
            continue
        image_name, response = line.split(",", 1)
        word = image_name.split(".")[0].strip()
        predicted_label = extract_word_type(response)
        if word in ground_truth:
            total_predictions += 1
            if predicted_label == ground_truth[word]:
                correct_predictions += 1
            else:
                if ground_truth[word] == 'a':
                    wrong_a += 1
                    print("truth is a->", response)
                if ground_truth[word] == 'b':
                    wrong_b += 1
                    print("truth is b->", response)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, correct_predictions, total_predictions, wrong_a, wrong_b

# ------------------ Hooked Noise ------------------
def apply_noise_with_mask(model, mask):
    layer_names = define_layers(model)
    hooks, index = [], -1
    for name, module in model.named_modules():
        if name in layer_names:
            index += 1
            if index == len(mask):
                break
            def hook_fn(module, input, output, mask=mask[index]):
                output_clone = output
                if isinstance(output, tuple):
                    output = output[0]
                device = output.device
                mask = mask.to(device)[:output.shape[1]]
                if output.ndim == 3:
                    adjusted_mask = mask.unsqueeze(0).unsqueeze(2)
                elif output.ndim == 2:
                    adjusted_mask = mask.unsqueeze(0)
                elif output.ndim == 4:
                    adjusted_mask = mask.view(1, -1, 1, 1)
                else:
                    raise ValueError(f"Unsupported output dimensions: {output.shape}")
                scale_factor = predefined_factor if predefined_factor != 0 else np.random.uniform(0, 1)
                if ZO:
                    scale_factor = 0
                inv_mask = output * (1 - adjusted_mask)
                scaled_output = inv_mask + output * adjusted_mask * scale_factor
                return (scaled_output, *output_clone[1:]) if isinstance(output_clone, tuple) else scaled_output
            hooks.append(module.register_forward_hook(hook_fn))
            print(f"Hook registered for {name}")
    print(f"Total layers: {len(layer_names)}, Hooks registered: {index + 1}")
    return hooks

hooks = apply_noise_with_mask(model, mask)

# ------------------ Inference ------------------
image_dir = "./KSCT/KSCT_v1_noCap"
image_filenames_All = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpeg')])
ground_truth_file = "./KSCT/KSCT_v1_keys.csv"
ground_truth = load_ground_truth(ground_truth_file)
prompts = load_prompts(ground_truth_file)
accuracies = []

lines = []
image_filenames = rand.sample(image_filenames_All, sample_number)
for image_file in tqdm(image_filenames, desc="Processing Images"):
    image = Image.open(os.path.join(image_dir, image_file))
    this_image = image_file.split(".")[0].strip()
    prompt = prompts[this_image]
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"In which picture {prompt}? After providing the reason, give your final answer in this format: 'The answer is picture a' or 'The answer is picture b'"}
        ]
    }]
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
    print(f"Image: {image_file}, Model Answer: {output_text[0]}")
    lines.append(f"{image_file}, {output_text[0]}\n")
accuracy, correct, total, wrong_a, wrong_b = calculate_accuracy(lines)
print(f"{accuracy}, ({correct}/{total})\n")
accuracies.append(accuracy)

# ------------------ Stats ------------------
acc = np.array(accuracies)
mean, std = np.mean(acc), np.std(acc, ddof=1)
sem = std / np.sqrt(len(acc))
confidence_interval = st.t.interval(0.95, len(acc) - 1, loc=mean, scale=sem)

print("Mean Accuracy: {:.4f}".format(mean))
with open(file_path, "a") as file:
    file.write(f"Accuracies of mask number{idx}: {acc}, wrong_a: {wrong_a}, wrong_b:{wrong_b}\n")

for hook in hooks:
    hook.remove()
