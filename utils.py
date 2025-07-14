# utils.py
import json
import openai
from typing import Iterator, Tuple
from datasets import load_dataset


# ----------------------------- OpenAI helper ----------------------------- #

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client


def gpt(prompt: str, *, model: str = "gpt-4o", temperature: float = 0.2,
        system: str | None = None) -> str:
    client = _get_client()
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()


# ----------------------------- File I/O helpers ----------------------------- #

def load_gaia() -> Iterator[Tuple[str, str]]:
    dataset = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
    for item in dataset:
        yield item["id"], item["task"]


def save_json(data: dict, filepath: str, ensure_ascii: bool = False) -> None:
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=ensure_ascii, indent=2)


def save_markdown(content: str, filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.strip())
