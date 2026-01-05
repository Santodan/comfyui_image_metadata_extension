import re
import os
from ..meta import MetaField
from ..formatters import calc_model_hash, calc_vae_hash, calc_lora_hash

# --- Existing Helpers (Restored) ---

try:
    from ..formatters import calc_clip_hash
except ImportError:
    def calc_clip_hash(name):
        return f"hash_for_{name}"

def get_model_name(node_id, obj, prompt, extra_data, outputs, input_data):
    mode = input_data[0].get("load_mode", ["full_checkpoint"])[0]
    key = "ckpt_name" if mode == "full_checkpoint" else "base_model"
    return input_data[0].get(key, [None])[0]

def get_model_hash(node_id, obj, prompt, extra_data, outputs, input_data):
    model_name = get_model_name(node_id, obj, prompt, extra_data, outputs, input_data)
    if model_name:
        return calc_model_hash(model_name)
    return None

def get_vae_name(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("vae_model", [None])[0]
    return None

def get_vae_hash(node_id, obj, prompt, extra_data, outputs, input_data):
    vae_name = get_vae_name(node_id, obj, prompt, extra_data, outputs, input_data)
    if vae_name:
        return calc_vae_hash(vae_name)
    return None

def get_clip_names(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        clip_names = []
        for key in ["clip_model_1", "clip_model_2", "clip_model_3"]:
            name = input_data[0].get(key, [None])[0]
            if name and name != "None":
                clip_names.append(name)
        return clip_names if clip_names else None
    return None

def get_clip_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    names = get_clip_names(node_id, obj, prompt, extra_data, outputs, input_data)
    if names:
        return [calc_clip_hash(name) for name in names]
    return None

def get_clip_type(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("clip_type", [None])[0]
    return None

def get_unet_dtype(node_id, obj, prompt, extra_data, outputs, input_data):
    if input_data[0].get("load_mode", ["full_checkpoint"])[0] == "separate_components":
        return input_data[0].get("weight_dtype", [None])[0]
    return None
    
def get_metadata_field(field_name, node_id, obj, prompt, extra_data, outputs, input_data):
    metadata_dict = input_data[0].get("metadata", [None])[0]
    if metadata_dict and isinstance(metadata_dict, dict):
        return metadata_dict.get(field_name)
    return None

# --- Hub Node Logic ---

def parse_lora_hub_data(input_data):
    """
    Parses all 'loras_X' inputs from the Hub node.
    Since these are inputs, input_data[0] contains the resolved strings.
    """
    all_loras = []
    inputs = input_data[0]
    
    # Check all possible lora string inputs on the hub
    for i in range(1, 5):
        lora_str = inputs.get(f"loras_{i}", "")
        if isinstance(lora_str, list): lora_str = lora_str[0]
        if lora_str and isinstance(lora_str, str):
            # Regex to find: "path/lora.safetensors (0.85)"
            matches = re.findall(r"([^,]+?)\s\(([-+]?\d*\.?\d+)\)", lora_str)
            for m in matches:
                all_loras.append({"name": m[0].strip(), "strength": float(m[1])})
    return all_loras

def get_hub_lora_names(node_id, obj, prompt, extra_data, outputs, input_data):
    data = parse_lora_hub_data(input_data)
    return [d["name"] for d in data] if data else None

def get_hub_lora_strengths(node_id, obj, prompt, extra_data, outputs, input_data):
    data = parse_lora_hub_data(input_data)
    return [d["strength"] for d in data] if data else None

def get_hub_lora_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    data = parse_lora_hub_data(input_data)
    if not data: return None
    hashes = []
    for d in data:
        try:
            # Full path is preserved now, so calc_lora_hash will work
            h = calc_lora_hash(d["name"], input_data)
            hashes.append(h if h else None)
        except:
            hashes.append(None)
    return hashes

# --- Mapping ---

CAPTURE_FIELD_LIST = {
    "ModelAssembler": {
        MetaField.MODEL_NAME: {"selector": get_model_name},
        MetaField.MODEL_HASH: {"selector": get_model_hash},
        MetaField.VAE_NAME: {"selector": get_vae_name},
        MetaField.VAE_HASH: {"selector": get_vae_hash},
        "Clip Model Name(s)": {"selector": get_clip_names},
        "Clip Model Hash(es)": {"selector": get_clip_hashes},
        "Clip Type": {"selector": get_clip_type},
        "UNet Weight Type": {"selector": get_unet_dtype},
    },
    "ModelAssemblerMetadata": {
        MetaField.MODEL_NAME:     {"selector": lambda *args: get_metadata_field("model_name", *args)},
        MetaField.MODEL_HASH:     {"selector": lambda *args: get_metadata_field("model_hash", *args)},
        MetaField.VAE_NAME:       {"selector": lambda *args: get_metadata_field("vae_name", *args)},
        MetaField.VAE_HASH:       {"selector": lambda *args: get_metadata_field("vae_hash", *args)},
        "Clip Model Name(s)": {"selector": lambda *args: get_metadata_field("clip_names", *args)},
        "Clip Model Hash(es)": {"selector": lambda *args: get_metadata_field("clip_hashes", *args)},
        "Clip Type":          {"selector": lambda *args: get_metadata_field("clip_type", *args)},
        "UNet Weight Type":   {"selector": lambda *args: get_metadata_field("unet_dtype", *args)},
    },
    "LoraMetadataHub": {
        MetaField.LORA_MODEL_NAME: {"selector": get_hub_lora_names},
        MetaField.LORA_MODEL_HASH: {"selector": get_hub_lora_hashes},
        MetaField.LORA_STRENGTH_MODEL: {"selector": get_hub_lora_strengths},
        MetaField.LORA_STRENGTH_CLIP: {"selector": get_hub_lora_strengths},
    }
}
