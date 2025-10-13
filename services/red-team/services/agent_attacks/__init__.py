"""
Agent-Specific Attack Vectors
Specialized attacks for LLM agents with tool use and multi-step reasoning
"""

from .tool_injection import ToolInjectionAttacks
from .prompt_leaking import PromptLeakingAttacks
from .recursive_attacks import RecursiveAttacks
from .chain_of_thought_attacks import ChainOfThoughtAttacks

__all__ = [
    'ToolInjectionAttacks',
    'PromptLeakingAttacks', 
    'RecursiveAttacks',
    'ChainOfThoughtAttacks'
]
