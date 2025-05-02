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
# FIX #1 : Warn if gradio / gradio_client versions mismatch
# --------------------------------------------------
try:
    import importlib.metadata as _im

    ver_gradio = _im.version("gradio").split(".")[0]
    import gradio_client

    ver_client = gradio_client.__version__.split(".")[0]
    if ver_gradio != ver_client:
        warnings.warn(
            f"[Gradio warning] gradio=={_im.version('gradio')}  "
            f"and gradio_client=={gradio_client.__version__} "
            "do not share the same major version. "
            "This is the known cause of 'argument of type bool is not iterable'. "
            "Pin them to matching versions to remove this warning."
        )
except Exception:
    pass
# --------------------------------------------------

# --------------------------------------------------
# FIX #2 : Monkey‑patch the buggy helper (fails when schema is bool)
# --------------------------------------------------
import gradio_client.utils as _gc_utils

if not hasattr(_gc_utils, "_orig_json_schema_to_python_type"):
    _gc_utils._orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):          # <-- illegal input
            # Return a dummy but valid type so downstream code keeps working
            return "object" if schema else "None"
        return _gc_utils._orig_json_schema_to_python_type(schema, defs)

    _gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
# --------------------------------------------------

# Allow running when the repo lives outside PYTHONPATH
if os.path.exists("__init__.py"):
    _spec = importlib.util.spec_from_file_location("scepter_ext", "__init__.py")
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules["scepter_ext"] = _pkg
    _spec.loader.exec_module(_pkg)

# ------------------  Inference imports (unchanged) -----------------
from inference.ace_plus_diffusers import ACEPlusDiffuserInference
from inference.utils import edit_preprocess
from examples.examples import all_examples

# ------------------  File‑system back‑ends (unchanged) -------------
inference_dict = {"ACE_DIFFUSER_PLUS": ACEPlusDiffuserInference}

fs_list = [
    Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
    Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
]
for one_fs in fs_list:
    FS.init_fs_client(one_fs)

csv.field_size_limit(sys.maxsize)

# emoji shortcuts (unchanged)
refresh_sty = "🔄"
clear_sty = "🗑️"
upload_sty = "🖼️"
sync_sty = "💾"
chat_sty = "💬"
video_sty = "🎥"

# ------------------------------------------------------------------
# --------------------------  UI CLASS  ----------------------------
# ------------------------------------------------------------------
lock = threading.Lock()


class DemoUI:
    def __init__(
        self,
        infer_dir: str = "./config/ace_plus_diffusers_infer.yaml",
        model_list: str = "./models/model_zoo.yaml",
    ):
        # ---------- model initialisation (unchanged) ----------
        self.model_yamls = [infer_dir]
        self.model_choices = {}
        self.default_model_name = ""
        for yaml_path in self.model_yamls:
            model_cfg = Config(load=True, cfg_file=yaml_path)
            model_name = model_cfg.NAME
            if model_cfg.IS_DEFAULT:
                self.default_model_name = model_name
            self.model_choices[model_name] = model_cfg
        if not self.default_model_name:
            self.default_model_name = next(iter(self.model_choices))
        self.model_name = self.default_model_name

        pipe_cfg = self.model_choices[self.default_model_name]
        infer_name = pipe_cfg.get("INFERENCE_TYPE", "ACE_DIFFUSER_PLUS")
        self.pipe = inference_dict[infer_name]()
        self.pipe.init_from_cfg(pipe_cfg)

        # ---------- downstream task & preprocessors (unchanged) ----------
        self.task_model_cfg = Config(load=True, cfg_file=model_list)
        self.task_model = {}
        self.task_model_list = []
        self.edit_type_dict = {"repainting": None}
        self.edit_type_list = ["repainting"]
        for task_name, task_model in self.task_model_cfg.MODEL.items():
            self.task_model[task_name.lower()] = task_model
            self.task_model_list.append(task_name.lower())
            for pp in task_model.get("PREPROCESSOR", []):
                if pp["TYPE"] not in self.edit_type_dict:
                    pp["REPAINTING_SCALE"] = task_model.get("REPAINTING_SCALE", 1.0)
                    self.edit_type_dict[pp["TYPE"]] = pp
        self.max_msgs = 20

        # ---------- example list (unchanged) ----------
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

    # ------------------------------------------------------------------
    #  All methods below are identical to your original implementation.
    #  (construct_edit_image, create_ui, set_callbacks, etc.)            |
    # ------------------------------------------------------------------
    #   ↓↓↓  paste the bodies from your original file here  ↓↓↓
    # ------------------------------------------------------------------
    def construct_edit_image(self, edit_image, edit_mask):
        if edit_image is not None and edit_mask is not None:
            edit_image_rgb = pillow_convert(edit_image, "RGB")
            edit_image_rgba = pillow_convert(edit_image, "RGBA")
            edit_mask = pillow_convert(edit_mask, "L")

            arr1 = np.array(edit_image_rgb)
            arr2 = np.array(edit_mask)[:, :, np.newaxis]
            result_array = np.concatenate((arr1, arr2), axis=2)
            layer = Image.fromarray(result_array)

            ret_data = {
                "background": edit_image_rgba,
                "composite": edit_image_rgba,
                "layers": [layer],
            }
            return ret_data
        else:
            return None

    # ↓↓ create_ui  (unchanged – omitted here for brevity) ↓↓
    #     … just paste your original definition …
    def create_ui(self):
        # (full body unchanged)
        with gr.Row(equal_height=True, visible=True):
            with gr.Column(scale=2):
                self.gallery_image = gr.Image(
                    height=600, interactive=False, type="pil", elem_id="Reference_image"
                )
            # ...   rest of UI definition identical to your file ...
            # -----------------------------------------------

    # ↓↓ set_callbacks (unchanged – paste original) ↓↓
    def set_callbacks(self):
        # full body identical to your original
        pass  # ← replace pass with your original callback code


# ------------------------------------------------------------------
# ---------------------------  DRIVER  -----------------------------
# ------------------------------------------------------------------
def run_gr(cfg: Config) -> None:
    with gr.Blocks() as demo:
        ui = DemoUI()
        ui.create_ui()        # FIX #4 – explicit call (your original had it)
        ui.set_callbacks()    # FIX #5 – explicit call (your original had it)

    # --------------------------------------------------------------
    # FIX #3 : fully disable API schema routes to avoid JSON‑Schema path
    # --------------------------------------------------------------
    gr.routes.app.api_info = lambda *a, **k: {}
    # --------------------------------------------------------------

    demo.launch(
        server_name="0.0.0.0",
        server_port=cfg.args.server_port,
        share=True,
        root_path=cfg.args.root_path,
        show_api=False,  # keeps FastAPI /docs disabled on ≥4.29
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argparser for Scepter demo")
    parser.add_argument(
        "--server_port", dest="server_port", type=int, default=2345, help="Port to serve on"
    )
    parser.add_argument("--root_path", dest="root_path", default="", help="(optional) sub‑path")
    cfg = Config(load=True, parser_ins=parser)
    run_gr(cfg)
