# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import csv
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

# Initialize filesystem clients
fs_list = [
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "ModelscopeFs",  "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs",        "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs",       "TEMP_DIR": "./cache"}, load=False),
]
for one_fs in fs_list:
    FS.init_fs_client(one_fs)

# Prevent CSV field overflow
csv.field_size_limit(sys.maxsize)

# Inference and examples imports
from inference.ace_plus_diffusers import ACEPlusDiffuserInference
from inference.utils import edit_preprocess
from examples.examples import all_examples

inference_dict = {
    "ACE_DIFFUSER_PLUS": ACEPlusDiffuserInference
}

lock = threading.Lock()

class DemoUI:
    def __init__(self,
                 infer_dir="./config/ace_plus_diffusers_infer.yaml",
                 model_list="./models/model_zoo.yaml"):
        # Load main model configs
        self.model_choices = {}
        self.default_model_name = ""
        for p in [infer_dir]:
            cfg = Config(load=True, cfg_file=p)
            self.model_choices[cfg.NAME] = cfg
            if cfg.IS_DEFAULT:
                self.default_model_name = cfg.NAME
        if not self.default_model_name:
            self.default_model_name = next(iter(self.model_choices))
        self.model_name = self.default_model_name
        main_cfg = self.model_choices[self.model_name]
        inf_type = main_cfg.get("INFERENCE_TYPE", "ACE")
        self.pipe = inference_dict[inf_type]()
        self.pipe.init_from_cfg(main_cfg)

        # Load task-specific LoRA models
        task_cfg = Config(load=True, cfg_file=model_list)
        self.task_model = {}
        self.task_model_list = []
        self.edit_type_dict = {"repainting": None}
        self.edit_type_list = ["repainting"]
        for tname, tcfg in task_cfg.MODEL.items():
            key = tname.lower()
            self.task_model[key] = tcfg
            self.task_model_list.append(key)
            for pre in tcfg.get("PREPROCESSOR", []):
                if pre["TYPE"] in self.edit_type_dict:
                    continue
                pre["REPAINTING_SCALE"] = tcfg.get("REPAINTING_SCALE", 1.0)
                self.edit_type_dict[pre["TYPE"]] = pre
                self.edit_type_list.append(pre["TYPE"])

        # Prepare examples
        self.all_examples = [
            [
                ex["task_type"], ex["edit_type"], ex["instruction"],
                ex["input_reference_image"], ex["input_image"],
                ex["input_mask"], ex["output_h"],
                ex["output_w"], ex["seed"]
            ]
            for ex in all_examples
        ]

    def construct_edit_image(self, img, mask):
        if img is None or mask is None:
            return None
        rgb  = pillow_convert(img, "RGB")
        rgba = pillow_convert(img, "RGBA")
        m    = pillow_convert(mask, "L")
        arr1 = np.array(rgb)
        arr2 = np.array(m)[:, :, None]
        layer = Image.fromarray(np.concatenate((arr1, arr2), axis=2))
        return {"background": rgba, "composite": rgba, "layers": [layer]}

    def create_ui(self):
        with gr.Row():
            with gr.Column(scale=2):
                self.output_image = gr.Image(type="pil", interactive=False, height=600)
            with gr.Column(scale=1) as self.side_panel:
                with gr.Accordion("Related Input Image", open=False):
                    self.pre_img  = gr.Image(type="pil", interactive=False, height=600)
                    self.pre_mask = gr.Image(type="pil", interactive=False, height=600)

        with gr.Row():
            self.model_dropdown = gr.Dropdown(
                choices=list(self.model_choices.keys()),
                value=self.default_model_name,
                label="Model Version"
            )
            self.task_dropdown = gr.Dropdown(
                choices=self.task_model_list,
                value=self.task_model_list[0],
                label="Task Type"
            )
            self.edit_dropdown = gr.Dropdown(
                choices=self.edit_type_list,
                value=self.edit_type_list[0],
                label="Edit Type"
            )

        self.log_box = gr.Markdown(label="System Log")

        with gr.Row():
            self.prompt_box   = gr.Textbox(
                placeholder='Type your instruction here',
                label="Instruction",
                lines=1
            )
            self.generate_btn = gr.Button("Generate", variant="primary")

        with gr.Accordion("Advanced", open=True):
            with gr.Row():
                self.ref_img_input  = gr.Image(type="pil", label="Reference Image", height=300)
                self.edit_img_input = gr.ImageMask(type="pil", label="Edit Image", height=300)
            with gr.Row():
                self.steps_slider   = gr.Slider(1, 1000, value=self.pipe.input.get("sample_steps", 20), label="Steps")
                self.scale_slider   = gr.Slider(1.0, 100.0, value=self.pipe.input.get("guide_scale", 4.5), label="Guidance Scale")
                self.seed_slider    = gr.Slider(-1, 10_000_000, value=-1, label="Seed")
            with gr.Row():
                self.h_slider       = gr.Slider(256, 1440, value=self.pipe.input.get("output_height", 1024), label="Height")
                self.w_slider       = gr.Slider(256, 1440, value=self.pipe.input.get("output_width", 1024), label="Width")
                self.repaint_slider = gr.Slider(0.0, 1.0, value=self.pipe.input.get("repainting_scale", 1.0), label="Repainting Scale")

        # Hidden for examples
        self.ex_edit_img  = gr.Image(type="pil", visible=False)
        self.ex_edit_mask = gr.Image(type="pil", image_mode="L", visible=False)

        self.examples = gr.Examples(
            fn=self.run_example,
            examples=self.all_examples,
            inputs=[
                self.task_dropdown, self.edit_dropdown, self.prompt_box,
                self.ref_img_input, self.ex_edit_img, self.ex_edit_mask,
                self.h_slider, self.w_slider, self.seed_slider
            ],
            outputs=[
                self.output_image, self.side_panel,
                self.pre_img, self.pre_mask,
                self.log_box, self.edit_img_input
            ],
            examples_per_page=6,
            run_on_click=True,
            cache_examples=False
        )

    def set_callbacks(self):
        def on_model_change(name):
            if name not in self.model_choices:
                return gr.update(value=self.model_name)
            if name != self.model_name:
                with lock:
                    del self.pipe
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    cfg = self.model_choices[name]
                    inf = cfg.get("INFERENCE_TYPE", "ACE")
                    self.pipe = inference_dict[inf]()
                    self.pipe.init_from_cfg(cfg)
                    self.model_name = name
            return (
                gr.update(value=name),
                gr.update(value=""),
                gr.update(value=self.pipe.input.get("sample_steps", 20)),
                gr.update(value=self.pipe.input.get("guide_scale", 4.5)),
                gr.update(value=self.pipe.input.get("output_height", 1024)),
                gr.update(value=self.pipe.input.get("output_width", 1024)),
                gr.update(value=self.pipe.input.get("repainting_scale", 1.0))
            )

        self.model_dropdown.change(
            on_model_change,
            inputs=[self.model_dropdown],
            outputs=[
                self.model_dropdown, self.prompt_box,
                self.steps_slider, self.scale_slider,
                self.h_slider, self.w_slider,
                self.repaint_slider
            ]
        )

        def on_task_change(task):
            proc = self.task_model[task].get("PREPROCESSOR", [])
            choices = ["repainting"] + [p["TYPE"] for p in proc]
            return gr.update(choices=choices, value=choices[0])

        self.task_dropdown.change(
            on_task_change,
            inputs=[self.task_dropdown],
            outputs=[self.edit_dropdown]
        )

        def on_edit_change(edit_type):
            info = (self.edit_type_dict.get(edit_type) or {})
            return gr.update(value=info.get("REPAINTING_SCALE", 1.0))

        self.edit_dropdown.change(
            on_edit_change,
            inputs=[self.edit_dropdown],
            outputs=[self.repaint_slider]
        )

        # <--- FIXED: outputs must be components, not dicts
        self.generate_btn.click(
            self.run_chat,
            inputs=[
                self.prompt_box,
                self.ref_img_input,
                self.edit_img_input,
                self.task_dropdown,
                self.edit_dropdown,
                self.scale_slider,
                self.steps_slider,
                self.seed_slider,
                self.h_slider,
                self.w_slider,
                self.repaint_slider
            ],
            outputs=[
                self.output_image,
                self.side_panel,
                self.pre_img,
                self.pre_mask,
                self.log_box
            ],
            queue=True
        )

    def preprocess_images(self, ref_img, edit_dict):
        if ref_img is not None:
            ref_img = pillow_convert(ref_img, "RGB")
        if edit_dict is None:
            return None, None, ref_img, True, ""
        bg   = edit_dict["background"]
        mask = np.array(edit_dict["layers"][0])[:, :, 3]
        if bg.sum() < 1 or mask.sum() < 1:
            return None, None, ref_img, False, "Please draw the mask region."
        return pillow_convert(bg, "RGB"), Image.fromarray(mask).convert("L"), ref_img, True, ""

    def run_chat(self, prompt,
                 ref_img, edit_dict,
                 task, edit_type,
                 scale, steps, seed,
                 height, width, repaint_scale):
        if task in ["portrait", "subject"] and ref_img is None:
            return None, gr.update(visible=True), None, None, "<mark>Provide reference image.</mark>"

        edit_img, edit_mask, ref_img, ok, err = self.preprocess_images(ref_img, edit_dict)
        if not ok:
            return None, gr.update(visible=True), None, None, f"<mark>{err}</mark>"

        pre = edit_preprocess(self.edit_type_dict[edit_type], we.device_id, edit_img, edit_mask)
        start = time.time()
        out_img, out_seed = self.pipe(
            reference_image  = ref_img,
            edit_image       = pre,
            edit_mask        = edit_mask,
            prompt           = prompt,
            output_height    = height,
            output_width     = width,
            sampler          = 'flow_euler',
            sample_steps     = steps,
            guide_scale      = scale,
            seed             = seed,
            repainting_scale = repaint_scale,
            lora_path        = self.task_model[task]["MODEL_PATH"]
        )
        dt = time.time() - start
        log = f"prompt: {prompt}; seed: {out_seed}; time: {dt:.2f}s"
        return out_img, gr.update(visible=True), pre or ref_img, edit_mask or None, log

    def run_example(self, task, edit, prompt,
                    ref_img, edit_img, edit_mask,
                    height, width, seed):
        card = self.construct_edit_image(edit_img, edit_mask)
        e_img, e_mask, ref_img, _, _ = self.preprocess_images(ref_img, card)
        pre = edit_preprocess(self.edit_type_dict[edit], we.device_id, e_img, e_mask)
        start = time.time()
        out_img, out_seed = self.pipe(
            reference_image  = ref_img,
            edit_image       = pre,
            edit_mask        = e_mask,
            prompt           = prompt,
            output_height    = height,
            output_width     = width,
            sampler          = 'flow_euler',
            sample_steps     = self.pipe.input.get("sample_steps", 20),
            guide_scale      = self.pipe.input.get("guide_scale", 4.5),
            seed             = seed,
            repainting_scale = self.edit_type_dict.get(edit, {}).get("REPAINTING_SCALE", 1.0),
            lora_path        = self.task_model[task]["MODEL_PATH"]
        )
        dt = time.time() - start
        log = f"prompt: {prompt}; seed: {out_seed}; time: {dt:.2f}s"
        composite = None
        if pre is not None and e_mask is not None:
            composite = Image.composite(Image.new("RGB", pre.size), pre, e_mask)
        return out_img, gr.update(visible=True), pre or ref_img, e_mask or None, log, gr.update(value=composite)

def run_gradio(cfg):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scepter Gradio Demo")
    parser.add_argument("--server_port", type=int, default=2345)
    parser.add_argument("--root_path",   type=str, default="")
    cfg = Config(load=True, parser_ins=parser)
    run_gradio(cfg)
