# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import csv
import importlib
import os
import sys
import threading
import time

import gradio as gr
import numpy as np
import torch
from PIL import Image

from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

# Allow importing when running from repo root
if os.path.exists('__init__.py'):
    pkg_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(pkg_name, '__init__.py')
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)

from inference.ace_plus_diffusers import ACEPlusDiffuserInference
from inference.utils import edit_preprocess
from examples.examples import all_examples

inference_dict = {"ACE_DIFFUSER_PLUS": ACEPlusDiffuserInference}

# Initialize FS backends
fs_backends = [
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "ModelscopeFs",  "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs",        "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs",       "TEMP_DIR": "./cache"}, load=False),
]
for fb in fs_backends:
    FS.init_fs_client(fb)

csv.field_size_limit(sys.maxsize)

# Emoji shortcuts
refresh_sty = '🔄'
clear_sty   = '🗑️'
upload_sty  = '🖼️'
sync_sty    = '💾'
chat_sty    = '💬'
video_sty   = '🎥'

lock = threading.Lock()

class DemoUI:
    def __init__(self,
                 infer_dir  = "./config/ace_plus_diffusers_infer.yaml",
                 model_list = './models/model_zoo.yaml'):
        # Load inference models
        self.model_yamls        = [infer_dir]
        self.model_choices      = {}
        self.default_model_name = ''
        for fn in self.model_yamls:
            cfg = Config(load=True, cfg_file=fn)
            name = cfg.NAME
            if cfg.IS_DEFAULT:
                self.default_model_name = name
            self.model_choices[name] = cfg
        if not self.default_model_name:
            self.default_model_name = next(iter(self.model_choices))
        self.model_name = self.default_model_name

        pipe_cfg   = self.model_choices[self.default_model_name]
        infer_type = pipe_cfg.get("INFERENCE_TYPE", "ACE_DIFFUSER_PLUS")
        self.pipe  = inference_dict[infer_type]()
        self.pipe.init_from_cfg(pipe_cfg)

        # Load task models and preprocessors
        task_cfg           = Config(load=True, cfg_file=model_list)
        self.task_model    = {}
        self.task_model_list = []
        self.edit_type_dict = {"repainting": None}
        self.edit_type_list = ["repainting"]
        for tname, tmod in task_cfg.MODEL.items():
            key = tname.lower()
            self.task_model[key] = tmod
            self.task_model_list.append(key)
            for pre in tmod.get("PREPROCESSOR", []):
                if pre["TYPE"] not in self.edit_type_dict:
                    pre["REPAINTING_SCALE"] = tmod.get("REPAINTING_SCALE", 1.0)
                    self.edit_type_dict[pre["TYPE"]] = pre
        self.max_msgs = 20

        # Prepare examples
        self.all_examples = [
            [
                ex["task_type"], ex["edit_type"], ex["instruction"],
                ex["input_reference_image"], ex["input_image"],
                ex["input_mask"], ex["output_h"], ex["output_w"], ex["seed"]
            ]
            for ex in all_examples
        ]

    def construct_edit_image(self, edit_image, edit_mask):
        if edit_image is not None and edit_mask is not None:
            rgb   = pillow_convert(edit_image, "RGB")
            rgba  = pillow_convert(edit_image, "RGBA")
            mask  = pillow_convert(edit_mask, "L")
            arr1  = np.array(rgb)
            arr2  = np.expand_dims(np.array(mask), axis=2)
            combined = np.concatenate((arr1, arr2), axis=2)
            layer    = Image.fromarray(combined)
            return {"background": rgba, "composite": rgba, "layers": [layer]}
        return None

    def create_ui(self):
        # ... paste your original full create_ui body here ...
        pass

    def set_callbacks(self):
        # ... paste your original full set_callbacks body here ...
        pass

def run_gr(cfg):
    with gr.Blocks() as demo:
        ui = DemoUI()
        ui.create_ui()
        ui.set_callbacks()

        # ───────────────────────────────────────────
        # ENABLE the async queue (restores upload buttons)
        # ───────────────────────────────────────────
        demo.queue()
        # ───────────────────────────────────────────

    demo.launch(
        server_name='0.0.0.0',
        server_port=cfg.args.server_port,
        share=True,
        root_path=cfg.args.root_path,
        show_api=False     # disables the buggy OpenAPI schema path
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scepter ACE LoRA demo')
    parser.add_argument('--server_port', type=int, default=2345)
    parser.add_argument('--root_path',    type=str, default='')
    cfg = Config(load=True, parser_ins=parser)
    run_gr(cfg)
