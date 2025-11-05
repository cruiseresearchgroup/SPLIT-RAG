from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
from openai import OpenAI

def _cut_at_stop(text: str, stop: Optional[List[str]]) -> str:
    if not stop: return text
    cut = len(text)
    for s in stop:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]

@dataclass
class VLLMConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    max_input_tokens: int = 4096

class VLLMProvider:
    """
    Uses OpenAI-compatible /completions endpoint provided by vLLM.
    """
    def __init__(self, cfg: VLLMConfig):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float,
                 top_p: float, stop: Optional[List[str]] = None) -> str:
        resp = self.client.completions.create(
            model=self.cfg.model,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=float(max(0.01, temperature)),
            top_p=float(top_p),
            stop=stop
        )
        text = resp.choices[0].text or ""
        return text

    def summarize(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        sp = "Summarize the following facts concisely without adding new facts.\n\n" + prompt
        return self.generate(sp, max_new_tokens, temperature, top_p=0.95, stop=["\n\n"])
