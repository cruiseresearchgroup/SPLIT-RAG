# src/splitrag/llm/hf_local.py
from __future__ import annotations
import threading, torch
from dataclasses import dataclass
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

def _cut_at_stop(text: str, stop: Optional[List[str]]) -> str:
    if not stop: 
        return text
    cut = len(text)
    for s in stop:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]

@dataclass
class HFLocalConfig:
    model_id: str = "tiiuae/Falcon-7B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_input_tokens: int = 2048
    max_concurrency: int = 1
    load_in_4bit: bool = False

class HFLocalProvider:
    """
    Local HuggingFace LLM provider using transformers.
    """
    def __init__(self, cfg: HFLocalConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        qargs = {
            "trust_remote_code": True,
        }
        if cfg.device == "cuda":
            qargs["torch_dtype"] = getattr(torch, cfg.dtype)
            qargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **qargs)
        self.model.eval()

        self._lock = threading.Semaphore(cfg.max_concurrency)

    def _encode(self, prompt: str, max_new_tokens: int):
        max_len = max(256, self.cfg.max_input_tokens - max_new_tokens - 8)
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
        return {k: v.to(self.model.device) for k, v in enc.items()}

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int, temperature: float,
                 top_p: float, stop: Optional[List[str]] = None) -> str:
        with self._lock:
            inputs = self._encode(prompt, max_new_tokens)
            out = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=float(max(0.01, temperature)),
                top_p=float(top_p),
                max_new_tokens=int(max_new_tokens),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return _cut_at_stop(text, stop)

    def summarize(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        sp = "Summarize the following facts concisely without adding new facts.\n\n" + prompt
        return self.generate(sp, max_new_tokens=max_new_tokens, temperature=temperature, top_p=0.95, stop=["\n\n"])
