# decomposition.py
import json
from typing import List

from models import SubTask
from utils import gpt


# ----------------------- Stage 1: Task decomposition ---------------------- #

DECOMPOSE_TEMPLATE = """
You are an AI research assistant. Decompose the following task into the smallest self‑contained sub‑tasks
that a team of autonomous agents could handle. Return a JSON list:

[
  {{ "id": "S1", "desc": "..." }},
  ...
]

Task:
{task}
"""


def decompose(task: str) -> List[SubTask]:
    json_txt = gpt(DECOMPOSE_TEMPLATE.format(task=task),
                  system="You are a helpful data‑only JSON generator.",
                  temperature=0)
    items = json.loads(json_txt)
    return [SubTask(id=item["id"], description=item["desc"]) for item in items]
