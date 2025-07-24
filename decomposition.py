# decomposition.py
import json
from typing import List

from models import SubTask
from utils import gpt
from prompts import DECOMPOSE_TEMPLATE


# ----------------------- Stage 1: Task decomposition ---------------------- #


def decompose(task: str) -> List[SubTask]:
    prompt = DECOMPOSE_TEMPLATE.format(task=task)
    print("\n--- DECOMPOSITION PROMPT ---")
    print(prompt)
    print("----------------------------")

    json_txt = gpt(prompt,
                  model="gpt-4o-2024-05-13",
                  system="You are a helpful data-only JSON generator. Respond with nothing but valid JSON.",
                  temperature=0)

    print("\n--- DECOMPOSITION LLM RESPONSE ---")
    print(json_txt)
    print("----------------------------------")
    
    # Clean the response from markdown code blocks
    if json_txt.startswith("```json"):
        json_txt = json_txt[7:-4]

    try:
        items = json.loads(json_txt)
        print(f"\nSuccessfully parsed {len(items)} subtasks.")
        return [SubTask(id=item["id"], description=item["desc"]) for item in items]
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {json_txt}")
        raise
