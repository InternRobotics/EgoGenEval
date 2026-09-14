#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_client.py

VLM client utilities shared by the SSP evaluator: object-anchor and
semantic annotation for the action-conditioned spatial benchmark.

Provides:
  - VLMClient: unified interface for OpenAI, local transformers, and dry-run backends.
  - call_with_retries: retry wrapper around VLMClient.call().
  - extract_json_object: robust JSON extraction from VLM text responses.
"""

import base64
import io
import json
import os
import re
import sys
import time
from json import JSONDecodeError
from typing import Any, Dict, List, Optional

from PIL import Image

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


def load_rgb_image(path: str) -> Optional[Image.Image]:
    if not path or not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def image_to_data_url(path: str, max_side: int = 1600, quality: int = 90) -> str:
    img = load_rgb_image(path)
    if img is None:
        raise FileNotFoundError(path)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class VLMResponseParseError(ValueError):
    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


def _balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _json_error_message(err: Exception, text: str) -> str:
    if isinstance(err, JSONDecodeError):
        start = max(0, err.pos - 240)
        end = min(len(text), err.pos + 240)
        excerpt = text[start:end].replace("\n", "\\n")
        return (
            f"Invalid JSON returned by VLM: {err.msg} at line {err.lineno}, "
            f"column {err.colno}, char {err.pos}. Around error: {excerpt}"
        )
    return f"Invalid JSON returned by VLM: {err}"


def _strip_json_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _coerce_json_object(value: Any, raw_text: str) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        dict_items = [item for item in value if isinstance(item, dict)]
        if len(dict_items) == 1:
            return dict_items[0]
        for item in dict_items:
            if "keep" in item:
                return item
    raise VLMResponseParseError(
        f"VLM returned JSON {type(value).__name__}, expected object. Raw response prefix: {raw_text[:500]}",
        raw_text,
    )


def extract_json_object(text: str) -> Dict[str, Any]:
    raw_text = text or ""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return _coerce_json_object(json.loads(text), raw_text)
    except VLMResponseParseError:
        raise
    except Exception as first_err:
        obj_text = _balanced_json_object(text)
        if obj_text:
            try:
                return _coerce_json_object(json.loads(obj_text), raw_text)
            except VLMResponseParseError:
                raise
            except Exception as second_err:
                repaired = _strip_json_trailing_commas(obj_text)
                if repaired != obj_text:
                    try:
                        return _coerce_json_object(json.loads(repaired), raw_text)
                    except VLMResponseParseError:
                        raise
                    except Exception:
                        pass
                raise VLMResponseParseError(_json_error_message(second_err, obj_text), raw_text) from second_err
        repaired = _strip_json_trailing_commas(text)
        if repaired != text:
            try:
                return _coerce_json_object(json.loads(repaired), raw_text)
            except VLMResponseParseError:
                raise
            except Exception:
                pass
        raise VLMResponseParseError(_json_error_message(first_err, text), raw_text) from first_err


def get_nested(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


class VLMClient:
    OPENAI_BACKENDS = {"openai_responses", "openai_chat", "openai_compatible_chat"}

    @staticmethod
    def _sanitize_ssl_env() -> None:
        cert_envs = {
            "SSL_CERT_FILE": os.path.isfile,
            "REQUESTS_CA_BUNDLE": os.path.isfile,
            "CURL_CA_BUNDLE": os.path.isfile,
            "SSL_CERT_DIR": os.path.isdir,
        }
        for key, validator in cert_envs.items():
            value = os.environ.get(key)
            if value and not validator(value):
                os.environ.pop(key, None)
                print(
                    f"[VLMClient] Ignored invalid {key}={value!r}; using default certificate paths.",
                    file=sys.stderr,
                )

    @staticmethod
    def _disable_proxy_env() -> None:
        for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]:
            if key in os.environ:
                os.environ.pop(key, None)
                print(f"[VLMClient] Ignored {key} for this VLM client run.", file=sys.stderr)

    def __init__(
        self,
        backend: str,
        model: str,
        api_key_env: str,
        base_url: Optional[str],
        temperature: float,
        max_output_tokens: int,
        timeout: float = 120.0,
        local_device_map: str = "auto",
        local_torch_dtype: str = "auto",
        local_attn_implementation: Optional[str] = None,
        local_min_pixels: Optional[int] = None,
        local_max_pixels: Optional[int] = None,
        disable_env_proxy: bool = False,
        json_response_format: bool = False,
    ):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.json_response_format = json_response_format
        self.client = None
        self.local_model = None
        self.local_processor = None
        self.local_torch = None

        if backend in self.OPENAI_BACKENDS:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package is not installed. Run: pip install -U openai")
            self._sanitize_ssl_env()
            if disable_env_proxy:
                self._disable_proxy_env()
            api_key = os.environ.get(api_key_env, "")
            if not api_key and backend in ["openai_responses", "openai_chat"]:
                raise EnvironmentError(f"Missing API key env var: {api_key_env}")
            kwargs = {"api_key": api_key or "EMPTY", "timeout": timeout}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)
        elif backend == "local_transformers":
            self._init_local_transformers(
                model_path=model,
                device_map=local_device_map,
                torch_dtype=local_torch_dtype,
                attn_implementation=local_attn_implementation,
                min_pixels=local_min_pixels,
                max_pixels=local_max_pixels,
            )
        elif backend == "dry_run":
            pass
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _resolve_torch_dtype(self, torch_module, dtype_name: str):
        name = (dtype_name or "auto").lower()
        if name == "auto":
            return "auto"
        mapping = {
            "bf16": torch_module.bfloat16,
            "bfloat16": torch_module.bfloat16,
            "fp16": torch_module.float16,
            "float16": torch_module.float16,
            "half": torch_module.float16,
            "fp32": torch_module.float32,
            "float32": torch_module.float32,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported --local_torch_dtype={dtype_name}. Use auto/bfloat16/float16/float32.")
        return mapping[name]

    def _load_local_model_class(self):
        import transformers
        class_names = [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ]
        for name in class_names:
            cls = getattr(transformers, name, None)
            if cls is not None:
                return cls
        raise ImportError("No suitable AutoModel class found in transformers. Please install a recent transformers version for Qwen3-VL.")

    def _init_local_transformers(
        self,
        model_path: str,
        device_map: str,
        torch_dtype: str,
        attn_implementation: Optional[str],
        min_pixels: Optional[int],
        max_pixels: Optional[int],
    ):
        try:
            import torch
            from transformers import AutoProcessor
        except Exception as e:
            raise ImportError(
                "local_transformers backend requires torch and transformers. "
                "For Qwen3-VL, use a recent transformers build that supports qwen3_vl_moe."
            ) from e

        model_cls = self._load_local_model_class()
        dtype = self._resolve_torch_dtype(torch, torch_dtype)
        processor_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = int(min_pixels)
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = int(max_pixels)
        try:
            self.local_processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        except TypeError:
            processor_kwargs.pop("min_pixels", None)
            processor_kwargs.pop("max_pixels", None)
            self.local_processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": device_map,
        }
        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = "auto"
        if attn_implementation and attn_implementation.lower() != "auto":
            model_kwargs["attn_implementation"] = attn_implementation

        try:
            self.local_model = model_cls.from_pretrained(model_path, **model_kwargs)
        except TypeError:
            # Some newer transformers releases renamed torch_dtype to dtype.
            if "torch_dtype" in model_kwargs:
                model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
            self.local_model = model_cls.from_pretrained(model_path, **model_kwargs)
        self.local_model.eval()
        self.local_torch = torch

    def _local_input_device(self):
        if self.local_model is None:
            return None
        device = getattr(self.local_model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        torch = self.local_torch
        if torch is not None and torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu") if torch is not None else None

    def _move_inputs_to_device(self, inputs: Any):
        device = self._local_input_device()
        if device is None:
            return inputs
        if hasattr(inputs, "to"):
            return inputs.to(device)
        torch = self.local_torch
        if isinstance(inputs, dict):
            return {k: (v.to(device) if torch is not None and torch.is_tensor(v) else v) for k, v in inputs.items()}
        return inputs

    def _local_inputs_have_vision_tensors(self, inputs: Any) -> bool:
        if isinstance(inputs, dict):
            keys = set(inputs.keys())
        elif hasattr(inputs, "keys"):
            keys = set(inputs.keys())
        else:
            keys = set()
        vision_keys = {
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "vision_infos",
        }
        return bool(keys & vision_keys)

    def _call_local_transformers(self, prompt: str, montage_path: str) -> Dict[str, Any]:
        if self.local_model is None or self.local_processor is None:
            raise RuntimeError("local_transformers backend was not initialized")
        image = load_rgb_image(montage_path)
        if image is None:
            raise FileNotFoundError(montage_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": montage_path},
                {"type": "text", "text": prompt},
            ],
        }]

        processor = self.local_processor
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            if not self._local_inputs_have_vision_tensors(inputs):
                raise RuntimeError("processor.apply_chat_template did not return image tensors")
        except Exception:
            if hasattr(processor, "apply_chat_template"):
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = prompt
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

        inputs = self._move_inputs_to_device(inputs)
        input_ids = inputs.get("input_ids") if isinstance(inputs, dict) else getattr(inputs, "input_ids", None)
        input_len = int(input_ids.shape[-1]) if input_ids is not None else 0

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_output_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature

        torch = self.local_torch
        with torch.inference_mode():
            generated_ids = self.local_model.generate(**inputs, **generate_kwargs)

        if input_len > 0 and getattr(generated_ids, "ndim", 0) == 2:
            generated_ids = generated_ids[:, input_len:]
        text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return extract_json_object(text)

    def call(self, prompt: str, montage_path: str) -> Dict[str, Any]:
        if self.backend == "dry_run":
            return {
                "keep": False,
                "reject_reason": "dry_run_no_vlm_call",
                "quality_filter": {
                    "image_quality": "unknown",
                    "blur_level": "unknown",
                    "exposure": "unknown",
                    "same_scene": "uncertain",
                    "motion_obviousness": "ambiguous",
                    "instruction_match": "uncertain",
                    "view_content_type": "normal_scene",
                    "has_meaningful_spatial_structure": False,
                    "has_clear_anchor_or_layout": False,
                    "over_cluttered": False,
                    "small_object_clutter": "none",
                    "fragmented_object_clutter": False,
                    "dominant_small_objects": False,
                    "anchor_trackability": "none",
                    "layout_occluded_by_clutter": False,
                    "confounders": []
                },
                "region_tags": {},
                "object_anchor_tags": {"detector_prompt_labels": []},
                "spatial_relation_tags": {},
                "difficulty_tags": [],
                "reason": "Dry run; no VLM was called."
            }

        if self.backend == "local_transformers":
            return self._call_local_transformers(prompt, montage_path)

        data_url = image_to_data_url(montage_path)
        if self.backend == "openai_responses":
            resp = self.client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ]}],
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            text = getattr(resp, "output_text", None)
            if text is None:
                parts = []
                for item in getattr(resp, "output", []) or []:
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", None) in ["output_text", "text"]:
                            parts.append(getattr(c, "text", ""))
                text = "\n".join(parts)
            return extract_json_object(text)

        if self.backend in ["openai_chat", "openai_compatible_chat"]:
            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]}],
                "temperature": self.temperature,
                "max_tokens": self.max_output_tokens,
            }
            if self.json_response_format:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(**kwargs)
            return extract_json_object(resp.choices[0].message.content)

        raise ValueError(f"Unknown backend: {self.backend}")


def call_with_retries(client: VLMClient, prompt: str, montage_path: str, max_retries: int, sleep: float) -> Dict[str, Any]:
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.call(prompt, montage_path)
        except Exception as e:
            last_err = e
            if attempt + 1 < max_retries:
                time.sleep(sleep * (attempt + 1))
    raise last_err
