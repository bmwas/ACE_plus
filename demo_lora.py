# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# --------------------------------------------------
#  ACE‑plus LoRA demo
# --------------------------------------------------

import argparse
import csv
import importlib
import os
import sys
import threading
import time
import warnings

import gradio as gr
import numpy as np
import torch
from PIL import Image

from scepter.modules.transform.io import pillow_convert
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

# --------------------------------------------------
# Warn if gradio / gradio_client versions mismatch
# --------------------------------------------------
try:
    import importlib.metadata as _im

    ver_gradio = _im.version("gradio").split(".")[0]
    ver_client = _im.version("gradio_client").split(".")[0]
    if ver_gradio != ver_client:
        warnings.warn(
            f"[Gradio warning] gradio and gradio_client major versions "
            f"do not match ({ver_gradio} vs {ver_client}). "
            "This is known to trigger 'TypeError: argument of type "
            "bool is not iterable' during OpenAPI generation. "
            "Pin them to the same release series to avoid surprises."
        )
except Exception:
    pass
# --------------------------------------------------

# Allow running when the repo lives outside PYTHONPATH
if os.path.exists("__init__.py"):
    _spec = importlib.util.spec_from_file_location("scepter_ext", "__init__.py")
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules["scepter_ext"] = _pkg
    _spec.loader.exec_module(_pkg)

# Local modules (unchanged)
from inference.ace_plus_diffusers import ACEPlusDiffuserInference
from inference.utils import edit_preprocess
from examples.examples import all_examples

# --------------------------------------------------
# File‑system back‑ends
# --------------------------------------------------
fs_list = [
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
]
for _fs in fs_list:
    FS.init_fs_client(_fs)

csv.field_size_limit(sys.maxsize)

# Emoji shortcuts
refresh_sty = "🔄"
clear_sty = "🗑️"
upload_sty = "🖼️"
sync_sty = "💾"
chat_sty = "💬"
video_sty = "🎥"

# Mapping from INFERENCE_TYPE to class
INFERENCE_DICT = {
    "ACE_DIFFUSER_PLUS": ACEPlusDiffuserInference,
}

lock = threading.Lock()


class DemoUI:
    """
    Build the Gradio interface and wire callbacks.
    """

    # --------------------------------------------------
    # Initialise the pipeline & task tables
    # --------------------------------------------------
    def __init__(
        self,
        infer_dir: str = "./config/ace_plus_diffusers_infer.yaml",
        model_list: str = "./models/model_zoo.yaml",
    ):
        # --- model yaml(s) ---------------------------------
        self.model_yamls = [infer_dir]
        self.model_choices = {}
        self.default_model_name = ""

        for yaml in self.model_yamls:
            cfg = Config(load=True, cfg_file=yaml)
            self.model_choices[cfg.NAME] = cfg
            if cfg.IS_DEFAULT:
                self.default_model_name = cfg.NAME

        if not self.default_model_name:
            self.default_model_name = next(iter(self.model_choices))

        # --- initialise inference pipeline -----------------
        self.model_name = self.default_model_name
        cfg0 = self.model_choices[self.model_name]
        infer_cls = INFERENCE_DICT[cfg0.get("INFERENCE_TYPE", "ACE_DIFFUSER_PLUS")]
        self.pipe = infer_cls()
        self.pipe.init_from_cfg(cfg0)

        # --- downstream task meta --------------------------
        self.task_model_cfg = Config(load=True, cfg_file=model_list)
        self.task_model = {}
        self.task_model_list = []
        self.edit_type_dict = {"repainting": None}
        self.edit_type_list = ["repainting"]

        for task_name, task_cfg in self.task_model_cfg.MODEL.items():
            tname = task_name.lower()
            self.task_model[tname] = task_cfg
            self.task_model_list.append(tname)
            for pp in task_cfg.get("PREPROCESSOR", []):
                if pp["TYPE"] not in self.edit_type_dict:
                    pp["REPAINTING_SCALE"] = task_cfg.get("REPAINTING_SCALE", 1.0)
                    self.edit_type_dict[pp["TYPE"]] = pp

        # --- re‑format examples ----------------------------
        self.all_examples = [
            [
                ex["task_type"],
                ex["edit_type"],
                ex["instruction"],
                ex["input_reference_image"],
                ex["input_image"],
                ex["input_mask"],
                ex["output_h"],
                ex["output_w"],
                ex["seed"],
            ]
            for ex in all_examples
        ]

    # --------------------------------------------------
    # Helper: merge edit image + mask into RGBA stack
    # --------------------------------------------------
    @staticmethod
    def construct_edit_image(edit_image, edit_mask):
        if edit_image is None or edit_mask is None:
            return None

        img_rgb = pillow_convert(edit_image, "RGB")
        img_rgba = pillow_convert(edit_image, "RGBA")
        mask = pillow_convert(edit_mask, "L")

        arr_rgb = np.asarray(img_rgb)
        arr_m = np.asarray(mask)[:, :, None]
        merged = np.concatenate([arr_rgb, arr_m], axis=2)
        layer = Image.fromarray(merged)

        return {"background": img_rgba, "composite": img_rgba, "layers": [layer]}

    # --------------------------------------------------
    # Build all UI components
    # --------------------------------------------------
    def create_ui(self):
        with gr.Row(equal_height=True):
            # ----- Left column : result gallery -----------------
            with gr.Column(scale=2):
                self.gallery_image = gr.Image(
                    height=600, interactive=False, type="pil", elem_id="Reference_image"
                )

            # ----- Right column : preprocess previews -----------
            with gr.Column(scale=1, visible=True) as self.edit_preprocess_panel:
                with gr.Row():
                    with gr.Accordion(label="Related Input Image", open=False):
                        self.edit_preprocess_preview = gr.Image(
                            height=600,
                            interactive=False,
                            type="pil",
                            elem_id="preprocess_image",
                        )
                        self.edit_preprocess_mask_preview = gr.Image(
                            height=600,
                            interactive=False,
                            type="pil",
                            elem_id="preprocess_image_mask",
                        )
                with gr.Row():
                    self.instruction = gr.Markdown(
                        """
**Instruction**  
1. Choose *Task Type* (portrait / subject / local editing).  
2. For ID‑preserving tasks supply a clear **Reference Image**.  
3. For local editing upload an **Edit Image**, draw a mask and select an
   *Edit Type*. Pre‑processing previews appear in “Related input image”.
"""
                    )

        # ----- model/task selectors ----------------------------
        with gr.Row():
            self.model_name_dd = gr.Dropdown(
                choices=list(self.model_choices),
                value=self.default_model_name,
                label="Model Version",
            )
            self.task_type = gr.Dropdown(
                choices=self.task_model_list,
                value=self.task_model_list[0],
                label="Task Type",
            )
            self.edit_type = gr.Dropdown(
                choices=self.edit_type_list, value=self.edit_type_list[0], label="Edit Type"
            )

        # ----- log panel ---------------------------------------
        with gr.Row():
            self.generation_info_preview = gr.Markdown(label="System Log")

        # ----- prompt + generate button ------------------------
        with gr.Row(variant="panel", equal_height=True):
            with gr.Column(scale=10):
                self.text = gr.Textbox(
                    placeholder='Type a prompt (use "@" to access history)',
                    label="Instruction",
                    lines=1,
                )
            with gr.Column(scale=2):
                self.chat_btn = gr.Button(value="Generate", variant="primary")

        # ----- advanced accordion ------------------------------
        with gr.Accordion(label="Advance", open=True):
            with gr.Row():
                with gr.Column():
                    self.reference_image = gr.Image(
                        height=1000,
                        interactive=True,
                        image_mode="RGB",
                        type="pil",
                        label="Reference Image",
                    )
                with gr.Column():
                    self.edit_image = gr.ImageMask(
                        height=1000,
                        interactive=True,
                        sources=["upload"],
                        type="pil",
                        label="Edit Image",
                        show_fullscreen_button=True,
                        format="png",
                    )
            with gr.Row():
                self.step = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=self.pipe.input.get("sample_steps", 20),
                    visible=self.pipe.input.get("sample_steps") is not None,
                    label="Sample Step",
                )
                self.cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=100.0,
                    value=self.pipe.input.get("guide_scale", 4.5),
                    visible=self.pipe.input.get("guide_scale") is not None,
                    label="Guidance Scale",
                )
                self.seed = gr.Slider(minimum=-1, maximum=10_000_000, value=-1, label="Seed")
                self.output_height = gr.Slider(
                    minimum=256,
                    maximum=1440,
                    value=self.pipe.input.get("output_height", 1024),
                    visible=self.pipe.input.get("output_height") is not None,
                    label="Output Height",
                )
                self.output_width = gr.Slider(
                    minimum=256,
                    maximum=1440,
                    value=self.pipe.input.get("output_width", 1024),
                    visible=self.pipe.input.get("output_width") is not None,
                    label="Output Width",
                )
                self.repainting_scale = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=self.pipe.input.get("repainting_scale", 1.0),
                    label="Repainting Scale",
                )
            with gr.Row():
                self.eg = gr.Column()

    # --------------------------------------------------
    # Wire all callbacks
    # --------------------------------------------------
    def set_callbacks(self):
        # ------------- change model ---------------------------
        def _on_model_change(model_name):
            if model_name not in self.model_choices:
                gr.Info("Invalid model name.")
                return gr.update()

            if model_name != self.model_name:
                with lock:
                    del self.pipe
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    cfg = self.model_choices[model_name]
                    cls = INFERENCE_DICT[cfg.get("INFERENCE_TYPE", "ACE_DIFFUSER_PLUS")]
                    self.pipe = cls()
                    self.pipe.init_from_cfg(cfg)
                    self.model_name = model_name

            return (
                model_name,
                gr.update(),
                gr.Slider(value=self.pipe.input.get("sample_steps", 20)),
                gr.Slider(value=self.pipe.input.get("guide_scale", 4.5)),
                gr.Slider(value=self.pipe.input.get("output_height", 1024)),
                gr.Slider(value=self.pipe.input.get("output_width", 1024)),
                gr.Slider(value=self.pipe.input.get("repainting_scale", 1.0)),
            )

        self.model_name_dd.change(
            _on_model_change,
            inputs=[self.model_name_dd],
            outputs=[
                self.model_name_dd,
                self.text,
                self.step,
                self.cfg_scale,
                self.output_height,
                self.output_width,
                self.repainting_scale,
            ],
        )

        # ------------- change task type -----------------------
        def _on_task_change(task_type):
            task_cfg = self.task_model[task_type]
            etypes = ["repainting"]
            for pp in task_cfg.get("PREPROCESSOR", []):
                pp["REPAINTING_SCALE"] = task_cfg.get("REPAINTING_SCALE", 1.0)
                self.edit_type_dict[pp["TYPE"]] = pp
                etypes.append(pp["TYPE"])
            return gr.update(choices=etypes, value=etypes[0])

        self.task_type.change(_on_task_change, inputs=self.task_type, outputs=self.edit_type)

        # ------------- change edit type -----------------------
        def _on_edit_change(edit_type):
            repaint = self.edit_type_dict[edit_type] or {}
            return gr.Slider(value=repaint.get("REPAINTING_SCALE", 1.0))

        self.edit_type.change(_on_edit_change, inputs=self.edit_type, outputs=self.repainting_scale)

        # ------------- common helper --------------------------
        def _preprocess_imgs(ref_img, edit_dict):
            if ref_img is not None:
                ref_img = pillow_convert(ref_img, "RGB")

            if edit_dict is None:
                return None, None, ref_img, True, ""

            edit_img = edit_dict["background"]
            edit_msk = np.asarray(edit_dict["layers"][0])[:, :, 3]

            if edit_img is None or np.sum(np.asarray(edit_img)) < 1:
                return None, None, ref_img, ref_img is not None, (
                    "You must provide a non‑empty Edit Image."
                )

            if np.sum(edit_msk) < 1:
                return None, None, ref_img, False, "You must draw the repainting mask."

            edit_img = pillow_convert(edit_img, "RGB")
            return edit_img, Image.fromarray(edit_msk).convert("L"), ref_img, True, ""

        # ------------- generate / chat ------------------------
        def _run(
            prompt,
            ref_img,
            edit_dict,
            task_type,
            edit_type,
            cfg_scale,
            step,
            seed,
            out_h,
            out_w,
            repaint_scale,
            _progress=gr.Progress(track_tqdm=True),
        ):
            model_path = self.task_model[task_type]["MODEL_PATH"]
            edit_info = self.edit_type_dict[edit_type]

            if task_type in ("portrait", "subject") and ref_img is None:
                return (
                    gr.Image(),
                    gr.Column(visible=True),
                    gr.Image(),
                    gr.Image(),
                    gr.Text(value="<mark>Please provide the reference image.</mark>"),
                )

            e_img, e_msk, r_img, ok, msg = _preprocess_imgs(ref_img, edit_dict)
            if not ok:
                return (
                    gr.Image(),
                    gr.Column(visible=True),
                    gr.Image(),
                    gr.Image(),
                    gr.Text(value=f"<mark>{msg}</mark>"),
                )

            e_img = edit_preprocess(edit_info, we.device_id, e_img, e_msk)
            t0 = time.time()
            img_out, seed = self.pipe(
                reference_image=r_img,
                edit_image=e_img,
                edit_mask=e_msk,
                prompt=prompt,
                output_height=out_h,
                output_width=out_w,
                sampler="flow_euler",
                sample_steps=step,
                guide_scale=cfg_scale,
                seed=seed,
                repainting_scale=repaint_scale,
                lora_path=model_path,
            )
            t1 = time.time()
            note = f"prompt: {prompt}; seed: {seed}; time: {t1-t0:.1f}s; repaint: {repaint_scale}"
            return (
                gr.Image(value=img_out),
                gr.Column(visible=True),
                gr.Image(value=e_img if e_img is not None else r_img),
                gr.Image(value=e_msk if e_msk is not None else None),
                gr.Text(value=note),
            )

        inps = [
            self.reference_image,
            self.edit_image,
            self.task_type,
            self.edit_type,
            self.cfg_scale,
            self.step,
            self.seed,
            self.output_height,
            self.output_width,
            self.repainting_scale,
        ]
        outs = [
            self.gallery_image,
            self.edit_preprocess_panel,
            self.edit_preprocess_preview,
            self.edit_preprocess_mask_preview,
            self.generation_info_preview,
        ]

        self.chat_btn.click(_run, inputs=[self.text] + inps, outputs=outs, queue=True)
        self.text.submit(_run, inputs=[self.text] + inps, outputs=outs, queue=True)

        # ------------- examples -------------------------------
        with self.eg:
            self.example_edit_image = gr.Image(visible=False)
            self.example_edit_mask = gr.Image(visible=False)
            gr.Examples(
                fn=_run,
                examples=self.all_examples,
                inputs=[
                    self.task_type,
                    self.edit_type,
                    self.text,
                    self.reference_image,
                    self.example_edit_image,
                    self.example_edit_mask,
                    self.output_height,
                    self.output_width,
                    self.seed,
                ],
                outputs=outs,
                examples_per_page=6,
                cache_examples=False,
                run_on_click=True,
            )


# --------------------------------------------------
# Driver
# --------------------------------------------------
def run_gr(cfg: Config):
    with gr.Blocks() as demo:
        ui = DemoUI()
        ui.create_ui()
        ui.set_callbacks()

    # ------------- SINGLE‑LINE FIX ------------------
    demo.launch(
        server_name="0.0.0.0",
        server_port=cfg.args.server_port,
        share=True,
        root_path=cfg.args.root_path,
        show_api=False,  # <-- prevents the JSON‑schema crash
    )
    # When you have matching wheel versions and want the
    # /docs endpoint back, simply set show_api=True.


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ACE‑plus LoRA Gradio demo")
    parser.add_argument("--server_port", type=int, default=2345)
    parser.add_argument("--root_path", default="")
    cfg = Config(load=True, parser_ins=parser)
    run_gr(cfg)
