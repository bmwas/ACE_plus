# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import random
from collections import OrderedDict

import torch, os
from diffusers import FluxFillPipeline
from scepter.modules.utils.config import Config
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger
from transformers import T5TokenizerFast
from .utils import ACEPlusImageProcessor

class ACEPlusDiffuserInference():
    def __init__(self, logger=None):
        if logger is None:
            logger = get_logger(name='ace_plus')
        self.logger = logger
        self.input = {}

    def load_default(self, cfg):
        if cfg is not None:
            self.input_cfg = {k.lower(): v for k, v in cfg.INPUT.items()}
            self.input = {k.lower(): dict(v).get('DEFAULT', None) if isinstance(v, (dict, OrderedDict, Config)) else v for k, v in cfg.INPUT.items()}
            self.output = {k.lower(): v for k, v in cfg.OUTPUT.items()}

    def init_from_cfg(self, cfg):
        self.max_seq_len = cfg.get("MAX_SEQ_LEN", 4096)
        self.image_processor = ACEPlusImageProcessor(max_seq_len=self.max_seq_len)

        local_folder = FS.get_dir_to_local_dir(cfg.MODEL.PRETRAINED_MODEL)

        self.pipe = FluxFillPipeline.from_pretrained(local_folder, torch_dtype=torch.bfloat16).to(we.device_id)

        tokenizer_2 = T5TokenizerFast.from_pretrained(os.path.join(local_folder, "tokenizer_2"),
                                                      additional_special_tokens=["{image}"])
        self.pipe.tokenizer_2 = tokenizer_2
        self.load_default(cfg.DEFAULT_PARAS)

    def prepare_input(self,
                      image,
                      mask,
                      batch_size=1,
                      dtype = torch.bfloat16,
                      num_images_per_prompt=1,
                      height=512,
                      width=512,
                      generator=None):
        num_channels_latents = self.pipe.vae.config.latent_channels
        # import pdb;pdb.set_trace()
        mask, masked_image_latents = self.pipe.prepare_mask_latents(
            mask.unsqueeze(0),
            image.unsqueeze(0).to(we.device_id, dtype = dtype),
            batch_size,
            num_channels_latents,
            num_images_per_prompt,
            height,
            width,
            dtype,
            we.device_id,
            generator,
        )
        # import pdb;pdb.set_trace()
        masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)
        return masked_image_latents

    @torch.no_grad()
    def __call__(self,
                 reference_image=None,
                 edit_image=None,
                 edit_mask=None,
                 prompt='',
                 task=None,
                 output_height=1024,
                 output_width=1024,
                 sampler='flow_euler',
                 sample_steps=28,
                 guide_scale=50,
                 lora_path=None,
                 seed=-1,
                 tar_index=0,
                 align=0,
                 repainting_scale=0,
                 **kwargs):
        if isinstance(prompt, str):
            prompt = [prompt]
        seed = seed if seed >= 0 else random.randint(0, 2 ** 32 - 1)
        
        # Debug print to help diagnose issues
        print(f"Reference image type before preprocessing: {type(reference_image)}")
        if reference_image is not None and hasattr(reference_image, 'size'):
            print(f"Reference image size before preprocessing: {reference_image.size}")

        # Branch 1: Simple text-to-image (no edit image, no reference image)
        if reference_image is None and edit_image is None:
            print("Text-to-image mode: no reference or edit image provided.")
            # Load LoRA weights if provided
            if lora_path is not None:
                with FS.get_from(lora_path) as local_path:
                    self.pipe.load_lora_weights(local_path)
            try:
                image = self.pipe(
                    prompt=prompt,
                    height=output_height,
                    width=output_width,
                    guidance_scale=guide_scale,
                    num_inference_steps=sample_steps,
                    max_sequence_length=512,
                    output_type="pil",
                    generator=torch.Generator("cpu").manual_seed(seed)
                ).images[0]
            finally:
                if lora_path is not None:
                    self.pipe.unload_lora_weights()
            return image, seed

        # Branch 2: Image editing WITHOUT reference image (edit image exists, reference image is None)
        elif reference_image is None and edit_image is not None:
            print("Edit mode: global edit, no reference image.")
            try:
                # Prepare a mask that covers the whole image if not provided
                if edit_mask is None:
                    if hasattr(edit_image, 'size'):
                        w, h = edit_image.size
                    else:
                        # Assume tensor [C, H, W]
                        h, w = edit_image.shape[-2:]
                    from PIL import Image as PILImage
                    edit_mask = PILImage.new("L", (w, h), 255)
                # Preprocess using ACEPlusImageProcessor
                image, mask, _, _, out_h, out_w, slice_w = self.image_processor.preprocess(
                    reference_image=edit_image,
                    edit_image=edit_image,
                    edit_mask=edit_mask,
                    width=output_width,
                    height=output_height,
                    repainting_scale=repainting_scale
                )
                h, w = image.shape[1:]
                generator = torch.Generator("cpu").manual_seed(seed)
                # Optionally load LoRA weights
                if lora_path is not None:
                    with FS.get_from(lora_path) as local_path:
                        self.pipe.load_lora_weights(local_path)
                # Patch the image processor to bypass internal preprocess
                original_processor = self.pipe.image_processor
                class DummyProcessor:
                    def __init__(self, orig_processor):
                        self.orig_processor = orig_processor
                    def preprocess(self, image, **kwargs):
                        return image
                    def postprocess(self, image, output_type=None):
                        return self.orig_processor.postprocess(image, output_type)
                self.pipe.image_processor = DummyProcessor(original_processor)
                try:
                    batch_image = image.unsqueeze(0)
                    batch_mask = mask.unsqueeze(0)
                    image = self.pipe(
                        prompt=prompt,
                        image=batch_image,
                        mask_image=batch_mask,
                        height=h,
                        width=w,
                        guidance_scale=guide_scale,
                        num_inference_steps=sample_steps,
                        max_sequence_length=512,
                        output_type="pil",
                        generator=generator
                    ).images[0]
                finally:
                    self.pipe.image_processor = original_processor
                if lora_path is not None:
                    self.pipe.unload_lora_weights()
                return self.image_processor.postprocess(image, slice_w, out_w, out_h), seed
            except Exception as e:
                print(f"Error in global edit pipeline: {str(e)}")
                import traceback
                traceback.print_exc()
                raise

        # Branch 3: Image editing WITH reference image (edit image and reference image provided)
        else:
            print("Edit mode: using edit_image and reference_image.")
            # Make sure we're working with PIL images for the processor
            try:
                # edit_image, edit_mask, change_image, content_image, out_h, out_w, slice_w
                image, mask, _, _, out_h, out_w, slice_w = self.image_processor.preprocess(reference_image, edit_image, edit_mask,
                                                                                    width = output_width,
                                                                                    height = output_height,
                                                                                    repainting_scale = repainting_scale)
                
                print(f"Image shape after ACE+ preprocessing: {image.shape if hasattr(image, 'shape') else 'Not a tensor'}")
                
                h, w = image.shape[1:]
                generator = torch.Generator("cpu").manual_seed(seed)
                masked_image_latents = self.prepare_input(image, mask,
                                                        batch_size=len(prompt), height=h, width=w, generator=generator)

                if lora_path is not None:
                    with FS.get_from(lora_path) as local_path:
                        self.pipe.load_lora_weights(local_path)
                        
                # wrapper to bypass internal preprocess but delegate output postprocess
                original_processor = self.pipe.image_processor
                class DummyProcessor:
                    def __init__(self, orig_processor):
                        self.orig_processor = orig_processor
                    def preprocess(self, image, **kwargs):
                        return image
                    def postprocess(self, image, output_type=None):
                        return self.orig_processor.postprocess(image, output_type)

                self.pipe.image_processor = DummyProcessor(original_processor)
                
                try:
                    # Prepare batch dims for pipeline
                    batch_image = image.unsqueeze(0)
                    batch_mask = mask.unsqueeze(0)
                    image = self.pipe(
                        prompt=prompt,
                        image=batch_image,
                        mask_image=batch_mask,
                        height=h,
                        width=w,
                        guidance_scale=guide_scale,
                        num_inference_steps=sample_steps,
                        max_sequence_length=512,
                        output_type="pil",
                        generator=generator
                    ).images[0]
                finally:
                    # Restore original processor
                    self.pipe.image_processor = original_processor
                    
                if lora_path is not None:
                    self.pipe.unload_lora_weights()
                    
                return self.image_processor.postprocess(image, slice_w, out_w, out_h), seed
                
            except Exception as e:
                print(f"Error in ACE+ diffuser pipeline: {str(e)}")
                import traceback
                traceback.print_exc()
                raise


if __name__ == '__main__':
    pass