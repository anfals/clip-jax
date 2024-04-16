# install open-clip
from absl import app
from typing import Sequence

import jax


from clip_jax.utils import load_config
from data import Dataset, image_to_logits, logits_to_image

from dataclasses import asdict, fields
from modeling import CLIPModel, CLIPTextTransformer, CLIPVisionTransformer
import json
import open_clip

def find_missing_keys(dict_a, dict_b, path=None):
    """
    Recursively finds keys in 'dict_a' absent in 'dict_b' and returns their full paths.

    Args:
        dict_a (dict): The primary dictionary.
        dict_b (dict): The dictionary to compare against.
        path (list, optional):  Builds up the path during recursion. Defaults to None.

    Returns:
        set: A set of the full paths to missing keys.
    """
    missing_keys = set()
    if path is None:
        path = []

    for key in dict_a:
        current_path = path + [key]
        if key not in dict_b:
            missing_keys.add(':'.join(current_path))
        elif isinstance(dict_a[key], dict) and isinstance(dict_b.get(key), dict):
            missing_keys.update(find_missing_keys(dict_a[key], dict_b.get(key), current_path))

    return missing_keys

def main(argv: Sequence[str]):
    # Get from huggingface; commenting out for now as I iterate
    # model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    # tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    # We need to align the configs so we initialize the model correctly
    clip_config = load_config("configs/large-patch16-clip.json")
    open_clip_hf_config = load_config("configs/open_clip_hf.json")

    # TODO: we need to go and figure out the name mismatches between open-jax and clip-jax 
    missing_keys = find_missing_keys(clip_config, open_clip_hf_config)
    print(missing_keys)

    # Here are the steps 
    # 1. Initialize the model
    # 2. A metric ton of manually copying things from the HF model to the params
    # 3. Use Orbax to save this Jax checkpoint 
    
    clip_model = CLIPModel(**clip_config)
    rng = jax.random.PRNGKey(0)
    model_inputs = clip_model.init_inputs(rng)
    params = clip_model.init(**model_inputs)["params"]

    a = 3



    


if __name__ == "__main__":
    app.run(main)