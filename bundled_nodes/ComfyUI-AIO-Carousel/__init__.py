"""Controls for AIO -> Qwen carousel. No diffusion/LLM weights are bundled."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

import folder_paths
import nodes

CATEGORY = "AIO Carousel"
NONE = "(none)"
PERSON = "Person"
DETAIL = "Detail - reference"
CROP = "Detail - crop anchor"


def image_choices():
    root = Path(folder_paths.get_input_directory())
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    return [NONE] + sorted(p.relative_to(root).as_posix() for p in root.rglob("*")
                           if p.is_file() and p.suffix.lower() in extensions)


def input_path(name):
    if not name or name == NONE:
        raise ValueError("Upload/select an image for this enabled input.")
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Image must be an existing file inside ComfyUI/input: " + str(name))
    return path


def image_hash(name):
    try:
        h = hashlib.sha256()
        with input_path(name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except ValueError:
        return "missing:" + str(name)


def load_image(name):
    with Image.open(input_path(name)) as pil:
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        return torch.from_numpy(np.asarray(pil, dtype=np.float32).copy() / 255.0)[None]


def scale_image(image, width, height, crop="disabled"):
    return nodes.ImageScale().upscale(image, "lanczos", int(width), int(height), crop)[0]


def work_size(width, height, megapixels):
    factor = min(1.0, math.sqrt(megapixels * 1_000_000 / (width * height)))
    return max(16, round(width * factor / 16) * 16), max(16, round(height * factor / 16) * 16)


def crop_bounds(width, height, x, y, crop_width, crop_height):
    left, top = int(x * width), int(y * height)
    right = min(width, max(left + 1, round((x + crop_width) * width)))
    bottom = min(height, max(top + 1, round((y + crop_height) * height)))
    return left, top, right, bottom


def expand_mask(mask, expand, feather):
    """Bounded PIL blur leaves exact zero outside its finite support."""
    pil = Image.fromarray(np.uint8(np.clip(mask, 0, 1) * 255), "L")
    if expand:
        pil = pil.filter(ImageFilter.MaxFilter(2 * int(expand) + 1))
    if feather:
        pil = pil.filter(ImageFilter.GaussianBlur(int(feather)))
    return np.asarray(pil, dtype=np.float32) / 255.0


class CarouselSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mode": (["Generate first with AIO", "Use approved first photo"],),
            "image": (image_choices(), {"image_upload": True}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "CAROUSEL_CONFIG")
    RETURN_NAMES = ("AIO_reference", "approved_photo", "source_config")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        try:
            input_path(image)
            return True
        except ValueError as exc:
            return str(exc)

    @classmethod
    def IS_CHANGED(cls, mode, image):
        return image_hash(image)

    def run(self, mode, image):
        picture = load_image(image)
        reuse = mode == "Use approved first photo"
        return (None if reuse else picture, picture if reuse else None,
                {"reuse": reuse, "enabled": True, "first": True, "input_file": image})


class CarouselSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "fixed_camera": ("BOOLEAN", {"default": False}),
            "match_color": ("BOOLEAN", {"default": True}),
            "color_strength": ("FLOAT", {"default": 0.25, "min": 0, "max": 1, "step": 0.05}),
            "upscale": ("BOOLEAN", {"default": True}),
            "edit_megapixels": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2, "step": 0.05}),
            "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
            "cfg": ("FLOAT", {"default": 4.0, "min": 1, "max": 10, "step": 0.1}),
            "save_diagnostics": ("BOOLEAN", {"default": True}),
            "mask_expand": ("INT", {"default": 32, "min": 0, "max": 128}),
            "mask_feather": ("INT", {"default": 12, "min": 0, "max": 64}),
            "segmentation_model": ("STRING", {"default": "yolov8n-seg.pt"}),
        }}

    RETURN_TYPES = ("CAROUSEL_CONFIG", "STRING", "BOOLEAN", "INT", "FLOAT")
    RETURN_NAMES = ("settings", "empty_prompt", "unfrozen", "steps", "cfg")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, **kwargs):
        return kwargs, "", False, kwargs["steps"], kwargs["cfg"]


class CarouselPromptControl:
    """Constant inputs: editing camera/quality controls must not invalidate the AIO generator."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("empty_prompt", "unfrozen")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self):
        return "", False


class CarouselAnchor:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_config": ("CAROUSEL_CONFIG",),
            "settings": ("CAROUSEL_CONFIG",),
            "generated": ("IMAGE", {"lazy": True}),
            "approved": ("IMAGE", {"lazy": True}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "CAROUSEL_CONFIG")
    RETURN_NAMES = ("full_anchor", "work_anchor", "first_config")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def check_lazy_status(self, source_config, settings, generated=None, approved=None):
        selected = "approved" if source_config["reuse"] else "generated"
        return [selected] if (approved if source_config["reuse"] else generated) is None else []

    def run(self, source_config, settings, generated=None, approved=None):
        full = approved if source_config["reuse"] else generated
        if full is None:
            raise ValueError("Selected anchor image is missing.")
        width, height = work_size(full.shape[2], full.shape[1], settings["edit_megapixels"])
        config = dict(settings, **source_config)
        return full, scale_image(full, width, height), config


def edit_instruction(config):
    if config["frame_mode"] == CROP:
        return "Crop of approved anchor; no LLM or diffusion model used."
    if config["frame_mode"] == DETAIL:
        task = (
            "Create a detail photograph. Image 2 defines the visible object, its arrangement and "
            "framing (food, drink, phone, jewelry or another detail). Do not force a whole person "
            "into this frame. Image 1 supplies the carousel's photographic finish and, only when "
            "the same object is actually visible there, its established design. The editor sees "
            "both images: refer to Image 1 and Image 2 explicitly. Never invent continuity for "
            "objects which cannot be seen. Preserve readable text from the relevant object. "
        )
    else:
        task = (
            "Image 1 is the approved photograph and authority for the same adult person's "
            "identity, facial/body proportions, hair, outfit, jewelry and physical location. "
            "Image 2 supplies the target pose, expression and observable framing only. "
            "Do not transfer its person's identity, body shape, clothes or lighting. "
        )
        task += (
            "The editor receives both images. Explicitly assign Image 1 to appearance and "
            "Image 2 only to pose; do not blend the people. " if config["guide_to_qwen"] else
            "The editor receives ONLY Image 1. Describe the visible target changes explicitly; "
            "do not refer to Image 2 or ask the editor to look at an unavailable guide. "
        )
        task += (
            "FIXED CAMERA: preserve the original camera, crop, background alignment and scale. "
            "Transfer the pose and expression within that frame; ignore guide camera changes. "
            if config["fixed_camera"] else
            "NEW VIEW ALLOWED: describe only clearly observable changes in framing, viewpoint "
            "and pose. Keep the same physical place; objects may shift naturally in perspective. "
        )
    task += (
        "Keep the first photograph's white balance, skin color, exposure, light direction, "
        "softness and shadow contrast, without HDR, skin smoothing or exaggerated pores/freckles. "
        "State visible hand/elbow positions, head tilt, gaze and mouth expression when a person "
        "is present. Never infer hidden limbs, invent camera measurements, lengthen a torso or "
        "change physical proportions. If uncertain, preserve the source. Copy clearly legible "
        "lettering exactly; do not guess obscured text. Preserve accessory count and design. "
        "Return one concise English edit instruction, about 90-140 words, with target changes "
        "first and a short preservation sentence. One photograph, no grid. "
    )
    if config.get("extra_instruction", "").strip():
        task += "User's additional direction: " + config["extra_instruction"].strip()
    return task


class CarouselSlide:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "settings": ("CAROUSEL_CONFIG",),
            "enabled": ("BOOLEAN", {"default": False}),
            "image": (image_choices(), {"image_upload": True}),
            "frame_mode": ([PERSON, DETAIL, CROP],),
            "camera": (["Global", "Fixed", "New angle"],),
            "guide_to_qwen": ("BOOLEAN", {"default": False}),
            "color": (["Global", "Off", "On"],),
            "seed": ("INT", {"default": 1235, "min": 0, "max": 0xffffffffffffffff}),
            "crop_x": ("FLOAT", {"default": 0, "min": 0, "max": 0.99, "step": 0.01}),
            "crop_y": ("FLOAT", {"default": 0, "min": 0, "max": 0.99, "step": 0.01}),
            "crop_width": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1, "step": 0.01}),
            "crop_height": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1, "step": 0.01}),
            "extra_instruction": ("STRING", {"default": "", "multiline": True}),
        }}

    RETURN_TYPES = ("CAROUSEL_CONFIG", "IMAGE", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("slide_config", "LLM_guide", "Qwen_optional_guide", "instruction", "seed")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def VALIDATE_INPUTS(cls, enabled, image, frame_mode):
        if not enabled or frame_mode == CROP:
            return True
        try:
            input_path(image)
            return True
        except ValueError as exc:
            return str(exc)

    @classmethod
    def IS_CHANGED(cls, enabled, image, frame_mode, **kwargs):
        return image_hash(image) if enabled and frame_mode != CROP else "no-file-needed"

    def run(self, settings, **kwargs):
        config = dict(settings, **kwargs)
        config["fixed_camera"] = (kwargs["camera"] == "Fixed" if kwargs["camera"] != "Global"
                                  else settings["fixed_camera"])
        if config["frame_mode"] != PERSON:
            config["fixed_camera"] = False
        config["match_color"] = (kwargs["color"] == "On" if kwargs["color"] != "Global"
                                 else settings["match_color"] and config["frame_mode"] == PERSON)
        if not kwargs["enabled"]:
            return config, None, None, "", kwargs["seed"]
        guide = None if kwargs["frame_mode"] == CROP else load_image(kwargs["image"])
        qwen_guide = guide if kwargs["frame_mode"] == DETAIL or kwargs["guide_to_qwen"] else None
        return config, guide, qwen_guide, edit_instruction(config), kwargs["seed"]


class CarouselRoute:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("CAROUSEL_CONFIG",), "anchor": ("IMAGE",),
            "edited": ("IMAGE", {"lazy": True}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def check_lazy_status(self, config, anchor, edited=None):
        return ["edited"] if config["enabled"] and config["frame_mode"] != CROP and edited is None else []

    def run(self, config, anchor, edited=None):
        if config["frame_mode"] != CROP:
            return (edited,)
        x, y, right, bottom = crop_bounds(anchor.shape[2], anchor.shape[1], config["crop_x"],
                                         config["crop_y"], config["crop_width"], config["crop_height"])
        crop = anchor[:, y:bottom, x:right, :]
        # Cover-crop preserves geometry; no anisotropic stretching of the detail.
        width, height = work_size(anchor.shape[2], anchor.shape[1], config["edit_megapixels"])
        return (scale_image(crop, width, height, "center"),)


_SEGMENTERS = {}


def main_person_mask(image, model_name):
    """Largest visible person only. Background people are not selected deliberately."""
    path = (Path(folder_paths.models_dir) / "ultralytics" / "segm" / model_name).resolve()
    root = (Path(folder_paths.models_dir) / "ultralytics" / "segm").resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Fixed camera needs models/ultralytics/segm/" + model_name +
                         ". Run the bundled installer with --models, or supply a manual mask.")
    from ultralytics import YOLO
    model = _SEGMENTERS.get(str(path))
    if model is None:
        model = _SEGMENTERS[str(path)] = YOLO(str(path))
    array = np.uint8(image[0].detach().cpu().numpy().clip(0, 1) * 255)
    result = model.predict(source=Image.fromarray(array), classes=[0], conf=0.25,
                           imgsz=1024, retina_masks=True, device="cpu", verbose=False)[0]
    if result.masks is None or not len(result.masks.data):
        raise ValueError("Fixed camera: no person mask found. Supply a manual old+new pose mask, "
                         "or turn Fixed camera OFF; the background was not silently unlocked.")
    masks = result.masks.data.detach().cpu().numpy()
    chosen = masks[np.argmax(masks.sum(axis=(1, 2)))]
    return np.asarray(Image.fromarray(np.uint8(chosen * 255)).resize(
        (image.shape[2], image.shape[1]), Image.Resampling.NEAREST), dtype=np.float32) / 255


def matched_color(image, anchor, strength):
    if strength == 0:
        return image
    from color_matcher import ColorMatcher
    reference = anchor[0].detach().cpu().numpy()
    output = []
    for frame in image.detach().cpu().numpy():
        matched = ColorMatcher().transfer(src=frame, ref=reference, method="reinhard")
        result = np.clip(frame + strength * (matched - frame), 0, 1)
        if not np.isfinite(result).all():
            raise ValueError("Color matching returned invalid pixels. Disable Match color and retry.")
        output.append(torch.from_numpy(result.astype(np.float32)))
    return torch.stack(output)


class CarouselFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("CAROUSEL_CONFIG",), "anchor": ("IMAGE",), "image": ("IMAGE",),
            "upscale_model": ("UPSCALE_MODEL", {"lazy": True}),
        }, "optional": {"manual_mask": ("MASK",)}}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("final_photo", "edited_area")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def check_lazy_status(self, config, anchor, image, upscale_model=None, manual_mask=None):
        return ["upscale_model"] if config["upscale"] and upscale_model is None else []

    def run(self, config, anchor, image, upscale_model=None, manual_mask=None):
        if image.shape[0] != 1 or anchor.shape[0] != 1:
            raise ValueError("Carousel processes one photograph per slide; batch_size must be 1.")
        raw = image
        working_anchor = scale_image(anchor, raw.shape[2], raw.shape[1])
        mask = None
        if config["fixed_camera"]:
            if manual_mask is not None:
                mask = manual_mask[0].detach().cpu().numpy()
                if np.max(mask) <= 0:
                    raise ValueError("Manual edit mask is empty; include both the old and new poses.")
                mask = np.asarray(Image.fromarray(np.uint8(np.clip(mask, 0, 1) * 255)).resize(
                    (raw.shape[2], raw.shape[1]), Image.Resampling.NEAREST), dtype=np.float32) / 255
            else:
                old = main_person_mask(working_anchor, config["segmentation_model"])
                new = main_person_mask(raw, config["segmentation_model"])
                mask = np.maximum(old, new)
            mask = expand_mask(mask, config["mask_expand"], config["mask_feather"])
        if config["match_color"] and config["frame_mode"] != CROP:
            image = matched_color(image, working_anchor, config["color_strength"])
        if config["upscale"]:
            image = nodes.NODE_CLASS_MAPPINGS["ImageUpscaleWithModel"]().upscale(upscale_model, image)[0]
        image = scale_image(image, anchor.shape[2], anchor.shape[1])
        if mask is None:
            final_mask = torch.ones((1, anchor.shape[1], anchor.shape[2]), dtype=torch.float32)
        else:
            resized = Image.fromarray(mask, "F").resize((anchor.shape[2], anchor.shape[1]), Image.Resampling.BILINEAR)
            final_mask = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy())[None].clamp(0, 1)
            # Composite LAST: upscaling/color correction cannot change protected pixels.
            image = anchor.cpu() * (1 - final_mask[..., None]) + image.cpu() * final_mask[..., None]
        return image, final_mask


class CarouselSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "config": ("CAROUSEL_CONFIG",), "image": ("IMAGE", {"lazy": True}),
            "filename_prefix": ("STRING", {"default": "AIO_Qwen_Carousel/slide_02"}),
        }, "optional": {
            "previous": ("STRING", {"forceInput": True}), "raw": ("IMAGE", {"lazy": True}),
            "mask": ("MASK", {"lazy": True}), "prompt_text": ("STRING", {"lazy": True, "forceInput": True}),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sequence_done",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def check_lazy_status(self, config, image=None, filename_prefix="", previous=None,
                          raw=None, mask=None, prompt_text=None, **kwargs):
        if not config.get("enabled", True):
            return []
        needed = ["image"] if image is None else []
        first = config.get("first", False)
        crop = config.get("frame_mode") == CROP
        if not (first and config.get("reuse")) and not crop and prompt_text is None:
            needed.append("prompt_text")
        if not first and config.get("save_diagnostics"):
            if raw is None:
                needed.append("raw")
            if config.get("fixed_camera") and mask is None:
                needed.append("mask")
        return needed

    def run(self, config, filename_prefix, image=None, previous=None, raw=None,
            mask=None, prompt_text=None, prompt=None, extra_pnginfo=None):
        if not config.get("enabled", True):
            return {"ui": {"images": []}, "result": (filename_prefix + ":OFF",)}
        if image is None:
            raise ValueError("Enabled carousel slide has no image.")
        info = dict(extra_pnginfo or {})
        info["carousel"] = {"settings": config, "edit_prompt": prompt_text or "",
                            "version": "AIO_QWEN_v1"}
        saved = nodes.SaveImage().save_images(image, filename_prefix, prompt, info)
        primary = saved["ui"]["images"]
        # Give prompt/settings the same counter as the PNG for reproducible comparison.
        for entry in primary:
            path = Path(folder_paths.get_output_directory()) / entry["subfolder"] / entry["filename"]
            path.with_suffix(".json").write_text(json.dumps(info["carousel"], indent=2, ensure_ascii=False), encoding="utf-8")
        if config.get("save_diagnostics") and not config.get("first"):
            if raw is not None:
                nodes.SaveImage().save_images(raw, filename_prefix + "_raw", prompt, info)
            if mask is not None and config.get("fixed_camera"):
                nodes.SaveImage().save_images(mask[..., None].repeat(1, 1, 1, 3), filename_prefix + "_mask", prompt, info)
        return {"ui": {"images": primary}, "result": (filename_prefix + ":DONE",)}


NODE_CLASS_MAPPINGS = {
    "AIOCarouselSource": CarouselSource,
    "AIOCarouselSettings": CarouselSettings,
    "AIOCarouselPromptControl": CarouselPromptControl,
    "AIOCarouselAnchor": CarouselAnchor,
    "AIOCarouselSlide": CarouselSlide,
    "AIOCarouselRoute": CarouselRoute,
    "AIOCarouselFinish": CarouselFinish,
    "AIOCarouselSave": CarouselSave,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIOCarouselSource": "01 First photo / AIO source",
    "AIOCarouselSettings": "Carousel controls",
    "AIOCarouselPromptControl": "Prompt cache controls",
    "AIOCarouselAnchor": "Approved anchor - full + work copy",
    "AIOCarouselSlide": "Slide reference - ON / OFF",
    "AIOCarouselRoute": "Edit or crop",
    "AIOCarouselFinish": "Match color / upscale / fixed background",
    "AIOCarouselSave": "Save enabled slide",
}
