
"""
Lightweight adaptation of TourSynbio/interface.py

Source:
https://github.com/tsynbio/TourSynbio

Adapted components:
1. GenerationConfig
2. generate_interactive interface

Modification:
Use Qwen2.5-1.5B through Ollama instead of
TourSynbio-7B through PyTorch/CUDA.
"""

from dataclasses import dataclass
from typing import Optional
import json
from unittest import result
import urllib.request


# Adapted from the original GenerationConfig
@dataclass
class GenerationConfig:
    max_length: Optional[int] = 30
    top_p: Optional[float] = 1.0
    temperature: Optional[float] = 0.0
    do_sample: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.0


# Adapted from the original generate_interactive
def generate_interactive(
    prompt,
    generation_config=None,
    model_name="qwen2.5:1.5b"
):
    if generation_config is None:
        generation_config = GenerationConfig()

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "num_predict": generation_config.max_length,
            "repeat_penalty": generation_config.repetition_penalty
        }
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(
        request,
        timeout=180
    ) as response:
        result = json.load(response)

    print("Prompt tokens:", result.get("prompt_eval_count"))
    print("Output tokens:", result.get("eval_count"))
    print("Finish reason:", result.get("done_reason"))

    return result["response"].strip()
