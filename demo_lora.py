# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import csv
import glob
import os
import sys
import threading
import time

import gradio as gr
import numpy as np
import torch, importlib
from PIL import Image
from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)

from inference.ace_plus_diffusers import ACEPlusDiffuserInference
from inference.utils import edit_preprocess
from examples.examples import all_examples

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

csv.field_size_limit(sys.maxsize)
refresh_sty = '\U0001f504'  # 🔄
clear_sty   = '\U0001f5d1'  # 🗑️
upload_sty  = '\U0001f5bc'  # 🖼️
sync_sty    = '\U0001f4be'  # 💾
chat_sty    = '\U0001F4AC'  # 💬
video_sty   = '\U0001f3a5'  # 🎥

lock = threading.Lock()

class DemoUI(object):
    def __init__(self,
                 infer_dir = "./config/ace_plus_diffusers_infer.yaml",
                 model_list='./models/model_zoo.yaml'
                 ):
        self.model_yamls = [infer_dir]
        self.model_choices = dict()
        self.default_model_name = ''
        for i in self.model_yamls:
            model_cfg = Config(load=True, cfg_file=i)
            model_name = model_cfg.NAME
            if model_cfg.IS_DEFAULT: self.default_model_name = model_name
            self.model_choices[model_name] = model_cfg
        print('Models: ', self.model_choices.keys())
        assert len(self.model_choices) > 0
        if self.default_model_name == "":
            self.default_model_name = list(self.model_choices.keys())[0]
        self.model_name = self.default_model_name
        pipe_cfg = self.model_choices[self.default_model_name]
        infer_name = pipe_cfg.get("INFERENCE_TYPE", "ACE")
        self.pipe = inference_dict[infer_name]()
        self.pipe.init_from_cfg(pipe_cfg)

        # task models
        self.task_model_cfg  = Config(load=True, cfg_file=model_list)
        self.task_model      = {}
        self.task_model_list = []
        self.edit_type_dict  = {"repainting": None}
        self.edit_type_list  = ["repainting"]
        for task_name, task_model in self.task_model_cfg.MODEL.items():
            tn = task_name.lower()
            self.task_model[tn]      = task_model
            self.task_model_list.append(tn)
            for pre in task_model.get("PREPROCESSOR", []):
                if pre["TYPE"] in self.edit_type_dict:
                    continue
                pre["REPAINTING_SCALE"] = task_model.get("REPAINTING_SCALE", 1.0)
                self.edit_type_dict[pre["TYPE"]] = pre

        # examples
        self.all_examples = [
            [
              ex["task_type"], ex["edit_type"], ex["instruction"],
              ex["input_reference_image"], ex["input_image"],
              ex["input_mask"], ex["output_h"],
              ex["output_w"], ex["seed"]
            ]
            for ex in all_examples
        ]

    def construct_edit_image(self, edit_image, edit_mask):
        if edit_image is not None and edit_mask is not None:
            edit_image_rgb  = pillow_convert(edit_image, "RGB")
            edit_image_rgba = pillow_convert(edit_image, "RGBA")
            edit_mask       = pillow_convert(edit_mask, "L")

            arr1 = np.array(edit_image_rgb)
            arr2 = np.array(edit_mask)[:, :, np.newaxis]
            result_array = np.concatenate((arr1, arr2), axis=2)
            layer = Image.fromarray(result_array)

            return {
                "background": edit_image_rgba,
                "composite":  edit_image_rgba,
                "layers":     [layer]
            }
        return None

    def create_ui(self):
        with gr.Row(equal_height=True, visible=True):
            with gr.Column(scale=2):
                self.gallery_image = gr.Image(
                    height=600, interactive=False, type='pil',
                    elem_id='Reference_image'
                )
            with gr.Column(scale=1, visible=True) as self.edit_preprocess_panel:
                with gr.Row():
                    with gr.Accordion(label='Related Input Image', open=False):
                        self.edit_preprocess_preview      = gr.Image(height=600, interactive=False, type='pil')
                        self.edit_preprocess_mask_preview = gr.Image(height=600, interactive=False, type='pil')
                with gr.Row():
                    instruction = """
**Instruction**:
1. Choose the Task Type (portrait / subject / local).
2. Upload a Reference Image for ID preservation.
3. Draw a mask on the Edit Image to specify the regeneration region.
4. For local editing, select an Edit Type.
"""
                    self.instruction = gr.Markdown(value=instruction)

        with gr.Row():
            self.model_name_dd = gr.Dropdown(
                choices=list(self.model_choices.keys()),
                value=self.default_model_name,
                label='Model Version'
            )
            self.task_type = gr.Dropdown(
                choices=self.task_model_list,
                value=self.task_model_list[0],
                label='Task Type'
            )
            self.edit_type = gr.Dropdown(
                choices=self.edit_type_list,
                value=self.edit_type_list[0],
                label='Edit Type'
            )

        with gr.Row():
            self.generation_info_preview = gr.Markdown(label='System Log.')

        with gr.Row(variant='panel', equal_height=True, show_progress=False):
            with gr.Column(scale=10, min_width=500):
                self.text = gr.Textbox(
                    placeholder='Input "@" to find history of image',
                    label='Instruction',
                    lines=1
                )
            with gr.Column(scale=2, min_width=100):
                self.chat_btn = gr.Button(value='Generate', variant="primary")

        with gr.Accordion(label='Advance', open=True):
            with gr.Row():
                with gr.Column():
                    self.reference_image = gr.Image(
                        height=1000, interactive=True,
                        image_mode='RGB', type='pil',
                        label='Reference Image'
                    )
                with gr.Column():
                    self.edit_image = gr.ImageMask(
                        height=1000, interactive=True,
                        value=None, sources=['upload'],
                        type='pil', layers=False,
                        label='Edit Image', format="png"
                    )
            with gr.Row():
                self.step = gr.Slider(
                    minimum=1, maximum=1000,
                    value=self.pipe.input.get("sample_steps", 20),
                    label='Sample Step'
                )
                self.cfg_scale = gr.Slider(
                    minimum=1.0, maximum=100.0,
                    value=self.pipe.input.get("guide_scale", 4.5),
                    label='Guidance Scale'
                )
                self.seed = gr.Slider(
                    minimum=-1, maximum=10_000_000,
                    value=-1, label='Seed'
                )
                self.output_height = gr.Slider(
                    minimum=256, maximum=1440,
                    value=self.pipe.input.get("output_height", 1024),
                    label='Output Height'
                )
                self.output_width = gr.Slider(
                    minimum=256, maximum=1440,
                    value=self.pipe.input.get("output_width", 1024),
                    label='Output Width'
                )
                self.repainting_scale = gr.Slider(
                    minimum=0.0, maximum=1.0,
                    value=self.pipe.input.get("repainting_scale", 1.0),
                    label='Repainting Scale'
                )

        with self.edit_preprocess_panel:
            # hidden example inputs for the Examples widget
            self.example_edit_image = gr.Image(type='pil', visible=False)
            self.example_edit_mask  = gr.Image(type='pil', image_mode='L', visible=False)

        self.examples = gr.Examples(
            fn=self.run_example,
            examples=self.all_examples,
            inputs=[
                self.task_type, self.edit_type, self.text,
                self.reference_image,
                self.example_edit_image, self.example_edit_mask,
                self.output_height, self.output_width, self.seed
            ],
            outputs=[
                self.gallery_image, self.edit_preprocess_panel,
                self.edit_preprocess_preview, self.edit_preprocess_mask_preview,
                self.generation_info_preview, self.edit_image
            ],
            examples_per_page=6,
            cache_examples=False,
            run_on_click=True
        )

    def set_callbacks(self):
        def change_model(model_name):
            if model_name not in self.model_choices:
                gr.Info('Invalid model name!')
                return gr.update(value=self.model_name)
            if model_name != self.model_name:
                with lock:
                    del self.pipe
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    cfg = self.model_choices[model_name]
                    infer_name = cfg.get("INFERENCE_TYPE", "ACE")
                    self.pipe = inference_dict[infer_name]()
                    self.pipe.init_from_cfg(cfg)
                    self.model_name = model_name
            return (
                gr.update(value=model_name),
                gr.update(value=""),
                gr.update(value=self.pipe.input.get("sample_steps", 20)),
                gr.update(value=self.pipe.input.get("guide_scale", 4.5)),
                gr.update(value=self.pipe.input.get("output_height", 1024)),
                gr.update(value=self.pipe.input.get("output_width", 1024)),
                gr.update(value=self.pipe.input.get("repainting_scale", 1.0))
            )

        self.model_name_dd.change(
            change_model,
            inputs=[self.model_name_dd],
            outputs=[
                self.model_name_dd, self.text,
                self.step, self.cfg_scale,
                self.output_height, self.output_width,
                self.repainting_scale
            ]
        )

        def change_task_type(task_type):
            task_info = self.task_model[task_type]
            new_edits = ["repainting"] + [
                p["TYPE"] for p in task_info.get("PREPROCESSOR", [])
            ]
            return gr.update(choices=new_edits, value=new_edits[0])

        self.task_type.change(
            change_task_type,
            inputs=[self.task_type],
            outputs=[self.edit_type]
        )

        def change_edit_type(edit_type):
            info = self.edit_type_dict.get(edit_type, {}) or {}
            scale = info.get("REPAINTING_SCALE", 1.0)
            return gr.update(value=scale)

        self.edit_type.change(
            change_edit_type,
            inputs=[self.edit_type],
            outputs=[self.repainting_scale]
        )

        self.chat_btn.click(
            self.run_chat,
            inputs=[
                self.text,
                self.reference_image,
                self.edit_image,
                self.task_type,
                self.edit_type,
                self.cfg_scale,
                self.step,
                self.seed,
                self.output_height,
                self.output_width,
                self.repainting_scale
            ],
            outputs=[
                self.gallery_image,
                self.edit_preprocess_panel,
                self.edit_preprocess_preview,
                self.edit_preprocess_mask_preview,
                self.generation_info_preview
            ],
            queue=True
        )

    def preprocess_input(self, ref_image, edit_image_dict):
        err_msg = ""
        if ref_image is not None:
            ref_image = pillow_convert(ref_image, "RGB")

        if edit_image_dict is None:
            edit_image = edit_mask = None
        else:
            edit_image = edit_image_dict["background"]
            edit_mask  = np.array(edit_image_dict["layers"][0])[:, :, 3]
            if edit_image.sum() < 1:
                edit_image = edit_mask = None
            elif edit_mask.sum() < 1:
                return None, None, None, False, "You must draw the repainting area."
            else:
                edit_image = pillow_convert(edit_image, "RGB")
                edit_mask  = Image.fromarray(edit_mask).convert('L')

        if ref_image is None and edit_image is None:
            return None, None, None, False, "Please provide the reference or edit image."
        return edit_image, edit_mask, ref_image, True, ""

    def run_chat(self, prompt,
                 ref_image, edit_image_dict,
                 task_type, edit_type,
                 cfg_scale, step, seed,
                 output_h, output_w, repainting_scale):
        # Validate reference if needed
        if task_type in ["portrait", "subject"] and ref_image is None:
            return None, gr.update(visible=True), None, None, "<mark>Please provide the reference image.</mark>"

        # Preprocess
        edit_image, edit_mask, ref_image, ok, err = self.preprocess_input(ref_image, edit_image_dict)
        if not ok:
            return None, gr.update(visible=True), None, None, f"<mark>{err}</mark>"

        # Apply preprocess
        pre_edit = edit_preprocess(self.edit_type_dict[edit_type], we.device_id, edit_image, edit_mask)

        # Inference
        start = time.time()
        out_image, out_seed = self.pipe(
            reference_image = ref_image,
            edit_image      = pre_edit,
            edit_mask       = edit_mask,
            prompt          = prompt,
            output_height   = output_h,
            output_width    = output_w,
            sampler         = 'flow_euler',
            sample_steps    = step,
            guide_scale     = cfg_scale,
            seed            = seed,
            repainting_scale=repainting_scale,
            lora_path       = self.task_model[task_type]["MODEL_PATH"]
        )
        dur = time.time() - start
        log = f"prompt: {prompt}; seed: {out_seed}; time: {dur:.2f}s"

        return (
            out_image,
            gr.update(visible=True),
            pre_edit or ref_image,
            edit_mask or None,
            log
        )

    def run_example(self, task_type, edit_type, prompt,
                    ref_image, edit_image, edit_mask,
                    output_h, output_w, seed):
        # same preprocessing as run_chat
        card = self.construct_edit_image(edit_image, edit_mask)
        edit_image, edit_mask, ref_image, _, _ = self.preprocess_input(ref_image, card)
        pre_edit = edit_preprocess(self.edit_type_dict[edit_type], we.device_id, edit_image, edit_mask)

        start = time.time()
        out_image, out_seed = self.pipe(
            reference_image = ref_image,
            edit_image      = pre_edit,
            edit_mask       = edit_mask,
            prompt          = prompt,
            output_height   = output_h,
            output_width    = output_w,
            sampler         = 'flow_euler',
            sample_steps    = self.pipe.input.get("sample_steps", 20),
            guide_scale     = self.pipe.input.get("guide_scale", 4.5),
            seed            = seed,
            repainting_scale = self.edit_type_dict.get(edit_type, {}).get("REPAINTING_SCALE", 1.0),
            lora_path       = self.task_model[task_type]["MODEL_PATH"]
        )
        dur = time.time() - start
        log = f"prompt: {prompt}; seed: {out_seed}; time: {dur:.2f}s"

        composite = None
        if pre_edit is not None and edit_mask is not None:
            composite = Image.composite(Image.new("RGB", pre_edit.size), pre_edit, edit_mask)

        return (
            out_image,
            gr.update(visible=True),
            pre_edit or ref_image,
            edit_mask or None,
            log,
            gr.update(value=composite)
        )

def run_gr(cfg):
    with gr.Blocks() as demo:
        ui = DemoUI()
        ui.create_ui()
        ui.set_callbacks()
        demo.launch(
            server_name='0.0.0.0',
            server_port=cfg.args.server_port,
            share=True,
            root_path=cfg.args.root_path
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argparser for Scepter:\n')
    parser.add_argument('--server_port', dest='server_port', type=int, default=2345)
    parser.add_argument('--root_path', dest='root_path', default='')
    cfg = Config(load=True, parser_ins=parser)
    run_gr(cfg)
