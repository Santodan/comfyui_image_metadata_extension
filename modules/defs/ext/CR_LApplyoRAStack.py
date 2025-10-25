# CR_ApplyLoRAStack.py
from ..meta import MetaField
from ..formatters import calc_lora_hash


def get_lora_stack_from_inputs(input_data):
    #print("[DEBUG] get_lora_stack_from_inputs been called")
    results = []
    for idx, item in enumerate(input_data):
        #print(f"[DEBUG] Processing input_data index {idx}: {item}")
        input_dict = item
        if "lora_stack" in input_dict:
            lora_stack = input_dict["lora_stack"]
            if lora_stack:
                for stack in lora_stack:
                    results.extend(stack)  # Flatten nested lists
    return results

    

def get_cr_lora_model_names(node_id, obj, prompt, extra_data, outputs, input_data):
    #print("[DEBUG] get_cr_lora_model_names called")
    lora_stack = get_lora_stack_from_inputs(input_data)
    return [entry[0] for entry in lora_stack]  # lora_name


def get_cr_lora_model_hashes(node_id, obj, prompt, extra_data, outputs, input_data):
    #print("[DEBUG] get_cr_lora_model_hashes called")
    lora_names = get_cr_lora_model_names(node_id, obj, prompt, extra_data, outputs, input_data)
    return [calc_lora_hash(model_name, input_data) for model_name in lora_names]


def get_cr_lora_strength_model(node_id, obj, prompt, extra_data, outputs, input_data):
    #print("[DEBUG] get_cr_lora_strength_model called")
    lora_stack = get_lora_stack_from_inputs(input_data)
    return [entry[1] for entry in lora_stack]  # model strength


def get_cr_lora_strength_clip(node_id, obj, prompt, extra_data, outputs, input_data):
    #print("[DEBUG] get_cr_lora_strength_clip called")
    lora_stack = get_lora_stack_from_inputs(input_data)
    return [entry[2] for entry in lora_stack]  # clip strength


CAPTURE_FIELD_LIST = {
    "CR Apply LoRA Stack": {
        MetaField.LORA_MODEL_NAME: {"selector": get_cr_lora_model_names},
        MetaField.LORA_MODEL_HASH: {"selector": get_cr_lora_model_hashes},
        MetaField.LORA_STRENGTH_MODEL: {"selector": get_cr_lora_strength_model},
        MetaField.LORA_STRENGTH_CLIP: {"selector": get_cr_lora_strength_clip},
    },
}
