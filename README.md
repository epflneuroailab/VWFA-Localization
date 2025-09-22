# VWFA-Localization

## Setup
1. First, download the **Qwen2VL-72B** model.
2. Place the model inside the `saved_model` directory.

## Scripts
- **extract_activations.py**  
  Records the activations of the model when presented with the localizer stimuli.

- **contrast_activations.py**  
  Computes the mean activations of units for words vs. other categories and generates VWFA masks.

- **apply_VWFA_mask_roar.py**  
  Applies the VWFA mask to the **ROAR** benchmark.

- **apply_VWFA_mask_KSCT.py**  
  Applies the VWFA mask to the **Kempler** benchmark.
