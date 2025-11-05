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
class OACompatConfig:
    base_url: str
    api_key: str
    model: str
    mode: str = "completions"  # "completions" or "chat"

class OACompatProvider:
    def __init__(self, cfg: OACompatConfig):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    def generate(self, prompt: str, max_new_tokens: int, temperature: float,
                 top_p: float, stop: Optional[List[str]] = None) -> str:
        if self.cfg.mode == "chat":
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "system", "content": "You are a helpful reasoning assistant."},
                          {"role": "user", "content": prompt}],
                temperature=float(max(0.01, temperature)),
                top_p=float(top_p),
                max_tokens=max_new_tokens,
                stop=stop
            )
            text = resp.choices[0].message.content or ""
        else:
            resp = self.client.completions.create(
                model=self.cfg.model,
                prompt=prompt,
                temperature=float(max(0.01, temperature)),
                top_p=float(top_p),
                max_tokens=max_new_tokens,
                stop=stop
            )
            text = resp.choices[0].text or ""
        return _cut_at_stop(text, stop)

    def summarize(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        sp = "Summarize the following facts concisely without adding new facts.\n\n" + prompt
        return self.generate(sp, max_new_tokens, temperature, top_p=0.95, stop=["\n\n"])
