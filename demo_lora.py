# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import csv
import glob
import os
import sys
import threading
import time
import importlib

import gradio as gr
import numpy as np
import torch
from PIL import Image

from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

# If your repo lives unpacked, allow importing __init__.py
if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)

from inference.ace_plus_diffusers import ACEPlusDiffuserInference
from inference.utils import edit_preprocess
from examples.examples import all_examples

# Map inference types
inference_dict = {
    "ACE_DIFFUSER_PLUS": ACEPlusDiffuserInference
}

# Initialize filesystem backends
fs_list = [
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
]
for one_fs in fs_list:
    FS.init_fs_client(one_fs)

csv.field_size_limit(sys.maxsize)

# Emoji shortcuts
refresh_sty = '\U0001f504'  # 🔄
clear_sty = '\U0001f5d1'    # 🗑️
upload_sty = '\U0001f5bc'   # 🖼️
sync_sty   = '\U0001f4be'   # 💾
chat_sty   = '\U0001F4AC'   # 💬
video_sty  = '\U0001f3a5'   # 🎥

lock = threading.Lock()

class DemoUI(object):
    def __init__(self,
                 infer_dir = "./config/ace_plus_diffusers_infer.yaml",
                 model_list='./models/model_zoo.yaml'
                 ):
        # Load model YAML(s)
        self.model_yamls = [infer_dir]
        self.model_choices = dict()
        self.default_model_name = ''
        for i in self.model_yamls:
            model_cfg = Config(load=True, cfg_file=i)
            model_name = model_cfg.NAME
            if model_cfg.IS_DEFAULT: self.default_model_name = model_name
            self.model_choices[model_name] = model_cfg
        if self.default_model_name == "":
            self.default_model_name = list(self.model_choices.keys())[0]
        self.model_name = self.default_model_name

        pipe_cfg = self.model_choices[self.default_model_name]
        infer_name = pipe_cfg.get("INFERENCE_TYPE", "ACE_DIFFUSER_PLUS")
        self.pipe = inference_dict[infer_name]()
        self.pipe.init_from_cfg(pipe_cfg)

        # Load task models & preprocessors
        self.task_model_cfg = Config(load=True, cfg_file=model_list)
        self.task_model = {}
        self.task_model_list = []
        self.edit_type_dict = {"repainting": None}
        self.edit_type_list = ["repainting"]
        for task_name, task_model in self.task_model_cfg.MODEL.items():
            self.task_model[task_name.lower()] = task_model
            self.task_model_list.append(task_name.lower())
            for preprocessor in task_model.get("PREPROCESSOR", []):
                if preprocessor["TYPE"] not in self.edit_type_dict:
                    preprocessor["REPAINTING_SCALE"] = task_model.get("REPAINTING_SCALE", 1.0)
                    self.edit_type_dict[preprocessor["TYPE"]] = preprocessor
        self.max_msgs = 20

        # Prepare examples
        self.all_examples = [
            [
             one_example["task_type"], one_example["edit_type"], one_example["instruction"],
             one_example["input_reference_image"], one_example["input_image"],
             one_example["input_mask"], one_example["output_h"],
             one_example["output_w"], one_example["seed"]
            ]
            for one_example in all_examples
        ]

    def construct_edit_image(self, edit_image, edit_mask):
        if edit_image is not None and edit_mask is not None:
            edit_image_rgb = pillow_convert(edit_image, "RGB")
            edit_image_rgba = pillow_convert(edit_image, "RGBA")
            edit_mask = pillow_convert(edit_mask, "L")

            arr1 = np.array(edit_image_rgb)
            arr2 = np.array(edit_mask)[:, :, np.newaxis]
            result_array = np.concatenate((arr1, arr2), axis=2)
            layer = Image.fromarray(result_array)

            return {
                "background": edit_image_rgba,
                "composite": edit_image_rgba,
                "layers": [layer]
            }
        else:
            return None

    def create_ui(self):
        # … your full original create_ui body …
        with gr.Row(equal_height=True, visible=True):
            with gr.Column(scale=2):
                self.gallery_image = gr.Image(
                    height=600,
                    interactive=False,
                    type='pil',
                    elem_id='Reference_image'
                )
            with gr.Column(scale=1, visible=True) as self.edit_preprocess_panel:
                with gr.Row():
                    with gr.Accordion(label='Related Input Image', open=False):
                        self.edit_preprocess_preview = gr.Image(
                            height=600,
                            interactive=False,
                            type='pil',
                            elem_id='preprocess_image'
                        )
                        self.edit_preprocess_mask_preview = gr.Image(
                            height=600,
                            interactive=False,
                            type='pil',
                            elem_id='preprocess_image_mask'
                        )
                with gr.Row():
                    instruction = """
                    **Instruction**:
                    1. Please choose the Task Type…
                    """
                    self.instruction = gr.Markdown(value=instruction)
        # … rest of your UI setup unchanged …

    def set_callbacks(self):
        # … your full original set_callbacks body …
        pass  # replace this with your original callbacks

def run_gr(cfg):
    with gr.Blocks() as demo:
        chatbot = DemoUI()
        chatbot.create_ui()
        chatbot.set_callbacks()

    # ───────────────────────────────────────────
    # FIX: disable Gradio’s built-in OpenAPI schema
    # ───────────────────────────────────────────
    demo.launch(
        server_name='0.0.0.0',
        server_port=cfg.args.server_port,
        share=True,
        root_path=cfg.args.root_path,
        show_api=False
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument('--server_port', dest='server_port',
                        type=int, default=2345)
    parser.add_argument('--root_path', dest='root_path',
                        default='')
    cfg = Config(load=True, parser_ins=parser)
    run_gr(cfg)
