"""Model backends. One vision-language model (Qwen2.5-VL) reads figures AND writes answers.

* HFBackend  - real model through transformers (Colab / Kaggle GPU). 4-bit by default so it fits a
               16 GB T4; on an A100 the 16 GB cap (CAP_GPU_16GB) keeps behaviour identical to the
               final Kaggle run.
* StubBackend - loads nothing and returns an empty string. ONLY for testing the plumbing (CSV
               format, retrieval, memory) on a machine without a GPU. Its answers are NOT real.

Both expose:  generate(prompt, image_paths, max_new_tokens) -> str   (greedy, deterministic)
"""
from __future__ import annotations

import logging

from .config import BACKEND, CAP_GPU_16GB, FIG_MAX_PIXELS, LOAD_4BIT, SEED, VLM_NAME

log = logging.getLogger("rag.llm")


class StubBackend:
    name = "stub"

    def generate(self, prompt: str, image_paths: list[str] | None = None, max_new_tokens: int = 256) -> str:
        return ""


class HFBackend:
    def __init__(self, name: str = VLM_NAME, load_4bit: bool = LOAD_4BIT):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        torch.manual_seed(SEED)
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. In Colab: Runtime > Change runtime type > GPU (A100/T4).")
        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        if CAP_GPU_16GB and total_gb > 17:
            # Pretend to be a 16 GB card so anything that works here also works on Kaggle T4/P100.
            torch.cuda.set_per_process_memory_fraction(16.0 / total_gb, 0)
            log.info("GPU has %.0f GB; capped to 16 GB to mimic the grading hardware", total_gb)

        self.name = name
        self.torch = torch
        # T4/P100 have no bfloat16; A100 does. Float16 is safe everywhere for inference.
        dtype = torch.float16
        kwargs = dict(torch_dtype=dtype, device_map={"": 0}, attn_implementation="sdpa")
        if load_4bit:
            try:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            except Exception as e:  # bitsandbytes missing -> caller should use the 3B model instead
                raise RuntimeError("bitsandbytes is required for 4-bit loading; `pip install bitsandbytes` "
                                   "or set VLM_NAME=Qwen/Qwen2.5-VL-3B-Instruct and LOAD_4BIT=0") from e
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(name, min_pixels=64 * 28 * 28, max_pixels=FIG_MAX_PIXELS)

    def generate(self, prompt: str, image_paths: list[str] | None = None, max_new_tokens: int = 256) -> str:
        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in (image_paths or [])]
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
        chat = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def load_backend():
    """Return the backend selected by RAG_BACKEND (default 'hf')."""
    if BACKEND == "stub":
        log.warning("STUB backend: no model is loaded; outputs are for plumbing tests only")
        return StubBackend()
    return HFBackend()
