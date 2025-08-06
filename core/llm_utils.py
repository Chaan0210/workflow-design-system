# core/llm_utils.py
from typing import Dict, Any, Optional, Callable
from utils import gpt, call_gpt_json
from prompts import PromptManager as CentralizedPromptManager

class LLMClient:
    """Unified LLM client with consistent error handling and retry logic."""
    
    def __init__(self, default_model: str = "gpt-4o-2024-05-13", default_temperature: float = 0.0):
        self.default_model = default_model
        self.default_temperature = default_temperature
    
    def call_text(self, prompt: str, system: str = "", model: Optional[str] = None, 
                  temperature: Optional[float] = None) -> str:
        """Make a text-based LLM call."""
        return gpt(
            prompt,
            model=model or self.default_model,
            system=system,
            temperature=temperature if temperature is not None else self.default_temperature
        )
    
    def call_json(self, prompt: str, system: str = "Return ONLY valid JSON. No prose.", 
                  validator: Optional[Callable] = None, model: Optional[str] = None,
                  temperature: Optional[float] = None) -> Dict[str, Any]:
        """Make a JSON-based LLM call with validation."""
        return call_gpt_json(
            prompt,
            system=system,
            validator=validator,
            model=model or self.default_model,
            temperature=temperature if temperature is not None else self.default_temperature
        )
    
    def call_with_fallback(self, prompt: str, fallback_data: Dict[str, Any], 
                          system: str = "Return ONLY valid JSON. No prose.",
                          validator: Optional[Callable] = None) -> Dict[str, Any]:
        """Make LLM call with fallback data if it fails."""
        try:
            return self.call_json(prompt, system, validator)
        except Exception as e:
            print(f"LLM call failed, using fallback: {str(e)}")
            return fallback_data


# Legacy PromptManager - now delegates to centralized prompts.py
class PromptManager:
    """Legacy prompt manager that delegates to centralized prompts.py"""
    
    @classmethod
    def format_dependency_prompt(cls, original_task: str, task_a: str, task_b: str) -> str:
        """Format dependency analysis prompt."""
        return CentralizedPromptManager.format_dependency_prompt(original_task, task_a, task_b)
    
    @classmethod
    def format_resource_conflict_prompt(cls, original_task: str, task_a: str, task_b: str) -> str:
        """Format resource conflict analysis prompt."""
        return CentralizedPromptManager.format_resource_conflict_prompt(original_task, task_a, task_b)
    
    @classmethod
    def format_decomposition_prompt(cls, task: str) -> str:
        """Format task decomposition prompt."""
        return CentralizedPromptManager.format_decomposition_prompt(task)
    
    @classmethod
    def format_mode_classification_prompt(cls, description: str, main_task: str) -> str:
        """Format mode classification prompt."""
        return CentralizedPromptManager.format_mode_classification_prompt(description, main_task)
    
    @classmethod
    def format_complexity_analysis_prompt(cls, task: str, subtasks: str) -> str:
        """Format complexity analysis prompt."""
        return CentralizedPromptManager.format_complexity_analysis_prompt(task, subtasks)
    
    @classmethod
    def format_quality_validation_prompt(cls, main_task: str, subtasks: str, dependencies: str) -> str:
        """Format quality validation prompt."""
        return CentralizedPromptManager.format_quality_validation_prompt(main_task, subtasks, dependencies)
