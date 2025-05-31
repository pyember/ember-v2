"""Judge functions for evaluating and synthesizing outputs.

Provides judge patterns for selecting or combining results.
"""

from typing import List, Any, Callable, Dict, Optional
from dataclasses import dataclass


@dataclass
class JudgeResult:
    """Result from a judge evaluation."""
    selected: Any
    reasoning: str
    scores: Optional[Dict[str, float]] = None


def create_selector_judge(model: Any) -> Callable:
    """Create a judge that selects the best option.
    
    Args:
        model: Language model to use for judging
        
    Returns:
        Judge function
    """
    def selector_judge(prompt: str, options: List[Any]) -> JudgeResult:
        """Select the best option given a prompt."""
        # Format options for the model
        options_text = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])
        
        judge_prompt = f"""Given the task: {prompt}
        
Select the best option:
{options_text}

Return the number of the best option and explain why."""
        
        response = model(judge_prompt).text
        
        # Simple parsing - in practice would be more robust
        try:
            # Extract number (assume it's the first digit)
            import re
            match = re.search(r'\d+', response)
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(options):
                    return JudgeResult(
                        selected=options[idx],
                        reasoning=response
                    )
        except:
            pass
            
        # Fallback to first option
        return JudgeResult(
            selected=options[0] if options else None,
            reasoning=response
        )
    
    return selector_judge


def create_synthesis_judge(model: Any) -> Callable:
    """Create a judge that synthesizes multiple inputs.
    
    Args:
        model: Language model to use for synthesis
        
    Returns:
        Synthesis function
    """
    def synthesis_judge(prompt: str, inputs: List[Any]) -> str:
        """Synthesize multiple inputs into a single output."""
        inputs_text = "\n".join([f"Input {i+1}: {inp}" for i, inp in enumerate(inputs)])
        
        synthesis_prompt = f"""Task: {prompt}

Synthesize these inputs into a single comprehensive response:
{inputs_text}"""
        
        response = model(synthesis_prompt).text
        return response.strip()
    
    return synthesis_judge