# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import glob
import io
import os
import numpy as np
from PIL import Image
from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS

from examples.examples import all_examples
from inference.ace_plus_diffusers import ACEPlusDiffuserInference
inference_dict = {
    "ACE_DIFFUSER_PLUS": ACEPlusDiffuserInference
}

fs_list = [
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
]

for one_fs in fs_list:
    FS.init_fs_client(one_fs)


def run_one_case(pipe,
                input_image = None,
                input_mask = None,
                input_reference_image = None,
                save_path = "examples/output/example.png",
                instruction = "",
                output_h = 1024,
                output_w = 1024,
                seed = -1,
                sample_steps = None,
                guide_scale = None,
                repainting_scale = None,
                model_path = None,
                **kwargs):
    # Added for debugging
    print("Input reference image path:", input_reference_image)
    
    print("\n--- DEBUG: Arguments to run_one_case ---")
    print(f"input_image: {input_image}")
    print(f"input_mask: {input_mask}")
    print(f"input_reference_image: {input_reference_image}")
    print(f"save_path: {save_path}")
    print(f"instruction: {instruction}")
    print(f"output_h: {output_h}, output_w: {output_w}")
    print(f"seed: {seed}")
    print(f"model_path: {model_path}")
    print("--------------------------------------\n")
    
    # Create a function to safely load images with better error handling
    def load_image_safely(image_path, mode="RGB"):
        if image_path is None:
            return None
            
        print(f"Loading image from: {image_path}")
        
        try:
            # First try direct file loading
            if os.path.exists(image_path):
                print(f"Loading using direct file path")
                return Image.open(image_path).convert(mode)
            else:
                # Fall back to the original method
                print(f"Loading using FS.get_object")
                image_data = FS.get_object(image_path)
                if image_data is None:
                    print(f"Error: FS.get_object returned None for {image_path}")
                    return None
                    
                img = Image.open(io.BytesIO(image_data))
                return pillow_convert(img, mode)
        except Exception as e:
            print(f"Error loading image '{image_path}': {e}")
            return None
    
    # Load all images
    input_image_pil = load_image_safely(input_image, "RGB") 
    input_mask_pil = load_image_safely(input_mask, "L")
    input_reference_image_pil = load_image_safely(input_reference_image, "RGB")
    print(f"Loaded input_image_pil: {input_image_pil}")
    print(f"Loaded input_mask_pil: {input_mask_pil}")
    print(f"Loaded input_reference_image_pil: {input_reference_image_pil}")
    
    # Check if images were loaded correctly
    if input_reference_image is not None and input_reference_image_pil is None:
        print(f"Failed to load reference image: {input_reference_image}")
        return None, seed
        
    # Print image types for debugging
    print(f"Reference image type: {type(input_reference_image_pil)}")
    if input_reference_image_pil is not None:
        print(f"Reference image size: {input_reference_image_pil.size}")
    
    # Call the pipeline
    try:
        print("Calling pipeline...")
        image, seed = pipe(
            reference_image=input_reference_image_pil,
            edit_image=input_image_pil,
            edit_mask=input_mask_pil,
            prompt=instruction,
            output_height=output_h,
            output_width=output_w,
            sampler='flow_euler',
            sample_steps=sample_steps or pipe.input.get("sample_steps", 28),
            guide_scale=guide_scale or pipe.input.get("guide_scale", 50),
            seed=seed,
            repainting_scale=repainting_scale or pipe.input.get("repainting_scale", 1.0),
            lora_path=model_path
        )
        print(f"Pipeline returned image: {image}, seed: {seed}")
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, seed
    
    # Make sure output directory exists
    print(f"Ensuring output directory exists: {os.path.dirname(save_path)}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the image
    try:
        print(f"Saving image to: {save_path}")
        with FS.put_to(save_path) as local_path:
            image.save(local_path)
        print(f"Image successfully saved to {local_path}")
        return local_path, seed
    except Exception as e:
        print(f"Error saving image: {e}")
        import traceback
        traceback.print_exc()
        return None, seed


def run():
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument('--instruction',
                        dest='instruction',
                        help='The instruction for editing or generating!',
                        default="")
    parser.add_argument('--output_h',
                        dest='output_h',
                        help='The height of output image for generation tasks!',
                        type=int,
                        default=1024)
    parser.add_argument('--output_w',
                        dest='output_w',
                        help='The width of output image for generation tasks!',
                        type=int,
                        default=1024)
    parser.add_argument('--input_reference_image',
                        dest='input_reference_image',
                        help='The input reference image!',
                        default=None
                        )
    parser.add_argument('--input_image',
                        dest='input_image',
                        help='The input image!',
                        default=None
                        )
    parser.add_argument('--input_mask',
                        dest='input_mask',
                        help='The input mask!',
                        default=None
                        )
    parser.add_argument('--save_path',
                        dest='save_path',
                        help='The save path for output image!',
                        default='examples/output_images/output.png'
                        )
    parser.add_argument('--seed',
                        dest='seed',
                        help='The seed for generation!',
                        type=int,
                        default=-1)

    parser.add_argument('--step',
                        dest='step',
                        help='The sample step for generation!',
                        type=int,
                        default=None)

    parser.add_argument('--guide_scale',
                        dest='guide_scale',
                        help='The guide scale for generation!',
                        type=int,
                        default=None)

    parser.add_argument('--repainting_scale',
                        dest='repainting_scale',
                        help='The repainting scale for content filling generation!',
                        type=int,
                        default=None)

    parser.add_argument('--task_type',
                        dest='task_type',
                        choices=['portrait', 'subject', 'local_editing'],
                        help="Choose the task type.",
                        default='')

    parser.add_argument('--task_model',
                        dest='task_model',
                        help='The models list for different tasks!',
                        default="./models/model_zoo.yaml")


    parser.add_argument('--infer_type',
                        dest='infer_type',
                        choices=['diffusers'],
                        default='diffusers',
                        help="Choose the inference scripts. 'native' refers to using the official implementation of ace++, "
                             "while 'diffusers' refers to using the adaptation for diffusers")

    parser.add_argument('--cfg_folder',
                        dest='cfg_folder',
                        help='The inference config!',
                        default="./config")

    cfg = Config(load=True, parser_ins=parser)

    model_yamls = glob.glob(os.path.join(cfg.args.cfg_folder, '*.yaml'))
    model_choices = dict()
    for i in model_yamls:
        model_cfg = Config(load=True, cfg_file=i)
        model_name = model_cfg.NAME
        model_choices[model_name] = model_cfg

    if cfg.args.infer_type == "native":
        infer_name = "ace_plus_native_infer"
    elif cfg.args.infer_type == "diffusers":
        infer_name = "ace_plus_diffuser_infer"
    else:
        raise ValueError("infer_type should be native or diffusers")

    assert infer_name in model_choices

    # choose different model
    task_model_cfg = Config(load=True, cfg_file=cfg.args.task_model)

    task_model_dict = {}
    for task_name, task_model in task_model_cfg.MODEL.items():
        task_model_dict[task_name] = task_model


    # choose the inference scripts.
    pipe_cfg = model_choices[infer_name]
    infer_name = pipe_cfg.get("INFERENCE_TYPE", "ACE_PLUS")
    pipe = inference_dict[infer_name]()
    pipe.init_from_cfg(pipe_cfg)

    if cfg.args.instruction == "" and cfg.args.input_image is None and cfg.args.input_reference_image is None:
        params = {
            "output_h": cfg.args.output_h,
            "output_w": cfg.args.output_w,
            "sample_steps": cfg.args.step,
            "guide_scale": cfg.args.guide_scale
        }
        # run examples

        for example in all_examples:
            example["model_path"] = FS.get_from(task_model_dict[example["task_type"].upper()]["MODEL_PATH"])
            example.update(params)
            if example["edit_type"] == "repainting":
                example["repainting_scale"] = 1.0
            else:
                example["repainting_scale"] = task_model_dict[example["task_type"].upper()].get("REPAINTING_SCALE", 1.0)
            print(example)
            local_path, seed = run_one_case(pipe, **example)

    else:
        print('DEBUG: task_model_cfg:', task_model_cfg)
        model_dict = task_model_cfg.MODEL
        if hasattr(model_dict, 'keys'):
            model_keys = list(model_dict.keys())
        else:
            model_keys = list(model_dict.__dict__.keys())
        print('DEBUG: task_model_cfg.MODEL keys:', model_keys)
        print('DEBUG: cfg.args.task_type.upper():', cfg.args.task_type.upper())
        assert cfg.args.task_type.upper() in model_keys
        params = {
            "input_image": cfg.args.input_image,
            "input_mask": cfg.args.input_mask,
            "input_reference_image": cfg.args.input_reference_image,
            "save_path": cfg.args.save_path,
            "instruction": cfg.args.instruction,
            "output_h": cfg.args.output_h,
            "output_w": cfg.args.output_w,
            "sample_steps": cfg.args.step,
            "guide_scale": cfg.args.guide_scale,
            "repainting_scale": cfg.args.repainting_scale,
            "model_path": FS.get_from(task_model_dict[cfg.args.task_type.upper()]["MODEL_PATH"])
        }
        local_path, seed = run_one_case(pipe, **params)
        print(local_path, seed)

if __name__ == '__main__':
    run()
