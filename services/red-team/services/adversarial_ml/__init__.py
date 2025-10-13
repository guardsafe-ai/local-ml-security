"""
Adversarial ML Attack Generation Module
State-of-the-art adversarial attack algorithms for LLM security testing
"""

from .gradient_attacks import GradientBasedAttacks
from .word_level_attacks import TextFoolerAttack, BERTAttack, HotFlipAttack
from .universal_triggers import UniversalTriggerGenerator
from .embedding_perturbation import EmbeddingPerturbation
from .multi_turn_attacks import MultiTurnAttacks, AttackType, AttackResult

__all__ = [
    'GradientBasedAttacks',
    'TextFoolerAttack', 
    'BERTAttack',
    'HotFlipAttack',
    'UniversalTriggerGenerator',
    'EmbeddingPerturbation',
    'MultiTurnAttacks',
    'AttackType',
    'AttackResult'
]
