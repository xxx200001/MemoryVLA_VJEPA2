import os
import numpy as np
import tensorflow as tf
import yaml
from argparse import Namespace

os.environ["DISPLAY"] = ""
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

from simpler_env.evaluation.argparse import get_args

from evaluation.simpler_env.maniskill2_evaluator import maniskill2_evaluator
from evaluation.simpler_env import VLAInference


def deep_update(base: dict, updates: dict):
    """Recursively update a dictionary with another dictionary."""
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


if __name__ == "__main__":
    args = get_args()

    with open(os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'config.yaml'), 'r') as f:
        yaml_args = yaml.safe_load(f) or {}

    cli_args = vars(args)
    merged_args = deep_update(yaml_args.copy(), cli_args)
    args = Namespace(**merged_args)

    exclude_keys = {"pretrained_checkpoint",}
    filtered_args = {k: v for k, v in vars(args).items() if k not in exclude_keys}

    args.logging_dir = os.path.join(os.path.dirname(os.path.dirname(args.ckpt_path)), 'eval_simpler')

    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    gpus = tf.config.list_physical_devices("GPU")

    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    assert args.ckpt_path is not None
    model = VLAInference(
        saved_model_path=args.ckpt_path,
        cfg_scale=1.5, # cfg from 1.5 to 7 also performs well
        **filtered_args,
    )

    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
