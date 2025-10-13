"""
Meta-Learning Module for ML Security
MAML-style meta-learning for rapid attack adaptation
"""

from .maml_attack_adaptation import MAMLAttackAdapter
from .few_shot_learning import FewShotAttackLearner
from .meta_optimizer import MetaOptimizer

__all__ = [
    'MAMLAttackAdapter',
    'FewShotAttackLearner',
    'MetaOptimizer'
]
