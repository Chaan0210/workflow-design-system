# core/llm_utils.py
import asyncio
from typing import Dict, Any, Optional, Callable, List, Tuple
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
    
    async def call_json_async(self, prompt: str, system: str = "Return ONLY valid JSON. No prose.", 
                             validator: Optional[Callable] = None, model: Optional[str] = None,
                             temperature: Optional[float] = None) -> Dict[str, Any]:
        """비동기 JSON LLM 호출"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.call_json(prompt, system, validator, model, temperature)
        )
    
    async def batch_dependency_analysis(self, dependency_pairs: List[Tuple], original_task: str = "", 
                                      batch_size: int = 10) -> Dict[Tuple[str, str], Tuple[bool, float]]:
        """병렬 의존성 분석 배치 처리 - Returns dict with (id_a, id_b) string tuples as keys"""
        results = {}
        
        # 배치 단위로 작업 분할
        for i in range(0, len(dependency_pairs), batch_size):
            batch = dependency_pairs[i:i + batch_size]
            print(f"   📤 Processing batch {i//batch_size + 1}/{(len(dependency_pairs)-1)//batch_size + 1} ({len(batch)} pairs)")
            
            # 배치 내 모든 요청을 병렬로 처리
            tasks = []
            for task_a, task_b in batch:
                prompt = CentralizedPromptManager.format_dependency_prompt(
                    original_task, task_a.description, task_b.description
                )
                tasks.append(self._analyze_dependency_async(prompt, (task_a, task_b)))
            
            # 병렬 실행
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 수집 - Use string ID tuples as keys for hashability
            for j, result in enumerate(batch_results):
                task_a, task_b = batch[j]
                pair_key = (task_a.id, task_b.id)  # Use string IDs as hashable keys
                
                if isinstance(result, Exception):
                    print(f"      ❌ Failed dependency check for {task_a.id} → {task_b.id}: {result}")
                    results[pair_key] = (False, 0.3)  # 폴백값
                else:
                    results[pair_key] = result
        
        return results
    
    async def _analyze_dependency_async(self, prompt: str, pair: Tuple) -> Tuple[bool, float]:
        """개별 의존성 분석 (비동기)"""
        try:
            data = await self.call_json_async(
                prompt, 
                validator=lambda d: self._validate_dependency_json(d)
            )
            return bool(data["dependent"]), float(data["confidence"])
        except Exception:
            return False, 0.3
    
    def _validate_dependency_json(self, d: dict) -> None:
        """의존성 JSON 검증"""
        if not isinstance(d, dict):
            raise ValueError(f"Expected dict, got {type(d)}")
        if "dependent" not in d:
            raise ValueError("Missing 'dependent' field")
        if "confidence" not in d:
            raise ValueError("Missing 'confidence' field")


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
