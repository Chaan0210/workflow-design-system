# utils.py
from __future__ import annotations
import json
import time
from typing import Callable, Optional, Any, Iterator, Tuple, Dict
import openai

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None  # datasets 미설치 환경에서도 임포트 에러 안나게

# ----------------------------- OpenAI client ----------------------------- #
_client: Optional[openai.OpenAI] = None

def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client

# ----------------------------- Basic GPT helper ----------------------------- #
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

# ----------------------------- Robust JSON GPT helper ----------------------------- #

RETRY_DEFAULTS = dict(max_retries=3, backoff=1.5)

def _strip_code_fence(txt: str) -> str:
    t = txt.strip()
    if t.startswith("```"):
        # 1) ```json ... ``` 혹은 ``` ... ``` 형태 모두 케어
        parts = t.split("```")
        # parts: ['', 'json\n{...}', ''] or ['', '{...}', '']
        # 가장 긴 JSON blob을 찾아서 반환
        # 단순화해서 가운데 조각을 선택 (len >= 2 가정)
        if len(parts) >= 2:
            candidate = parts[1]
            # "json\n{...}" 처럼 시작할 수 있으므로 첫 '{'부터 잘라냄
            if "{" in candidate:
                candidate = candidate[candidate.index("{"):]
            return candidate.strip()
    return t

def call_gpt_json(prompt: str,
                  *,
                  system: Optional[str] = None,
                  model: str = "gpt-4o",
                  temperature: float = 0.0,
                  validator: Optional[Callable[[dict], None]] = None,
                  max_retries: int = RETRY_DEFAULTS["max_retries"],
                  backoff: float = RETRY_DEFAULTS["backoff"]) -> dict:
    """
    LLM을 호출하여 **반드시 JSON**을 받는다.
    - code fence 제거
    - json.loads 실패 시 지수 백오프로 재시도
    - validator 제공 시 schema 검증 실패 시 재시도
    """
    client = _get_client()
    messages = [{"role": "user", "content": prompt}]
    if system:
        messages.insert(0, {"role": "system", "content": system})

    last_err: Optional[Exception] = None
    for i in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages
            )
            raw = resp.choices[0].message.content.strip()
            raw = _strip_code_fence(raw)
            data = json.loads(raw)
            if validator:
                validator(data)
            return data
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)

    raise RuntimeError(f"LLM JSON parsing failed after {max_retries} retries: {last_err}")

# ----------------------------- File / dataset helpers ----------------------------- #

def load_gaia(split: str = "2023_level1", part: str = "validation") -> Iterator[Tuple[str, str]]:
    """
    GAIA benchmark에서 (task_id, Question)을 yield.
    datasets가 설치되어 있지 않으면 RuntimeError 발생.
    """
    if load_dataset is None:
        raise RuntimeError("`datasets` 패키지가 설치되어 있지 않습니다. `pip install datasets` 하세요.")
    dataset = load_dataset("gaia-benchmark/GAIA", split=part, name=split)
    for item in dataset:
        yield item["task_id"], item["Question"]

def save_json(data: Any, filepath: str, ensure_ascii: bool = False) -> None:
    with open(filepath, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=ensure_ascii, indent=2)

def save_markdown(content: str, filepath: str) -> None:
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content.strip())
