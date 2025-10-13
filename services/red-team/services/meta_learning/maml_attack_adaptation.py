"""
MAML (Model-Agnostic Meta-Learning) for Attack Adaptation
Implements MAML algorithm for rapid adaptation to new attack patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


class MAMLAttackAdapter:
    """
    MAML-based attack adapter for rapid adaptation to new attack patterns
    Implements the MAML algorithm (Finn et al., 2017) for few-shot attack learning
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 inner_lr: float = 0.01,
                 meta_lr: float = 0.001,
                 num_inner_steps: int = 5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize MAML attack adapter
        
        Args:
            base_model: Base model to adapt
            inner_lr: Learning rate for inner loop (task-specific adaptation)
            meta_lr: Learning rate for outer loop (meta-optimization)
            num_inner_steps: Number of inner loop steps
            device: Device to run on
        """
        self.base_model = base_model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.device = device
        
        # Move model to device
        self.base_model.to(device)
        
        # Initialize meta-optimizer
        self.meta_optimizer = optim.Adam(self.base_model.parameters(), lr=meta_lr)
        
        # Attack pattern memory for few-shot learning
        self.attack_memory = defaultdict(list)
        self.adaptation_history = []
        
        logger.info(f"✅ MAML Attack Adapter initialized: inner_lr={inner_lr}, meta_lr={meta_lr}")
    
    def add_attack_patterns(self, 
                          attack_category: str, 
                          patterns: List[Dict[str, Any]],
                          labels: List[int]) -> None:
        """
        Add attack patterns to memory for few-shot learning
        
        Args:
            attack_category: Category of attacks
            patterns: List of attack patterns
            labels: Corresponding labels (0=safe, 1=attack)
        """
        try:
            for pattern, label in zip(patterns, labels):
                self.attack_memory[attack_category].append({
                    'pattern': pattern,
                    'label': label,
                    'timestamp': np.datetime64('now')
                })
            
            logger.info(f"Added {len(patterns)} patterns for category: {attack_category}")
            
        except Exception as e:
            logger.error(f"Failed to add attack patterns: {e}")
    
    def meta_train(self, 
                  support_sets: List[Dict[str, Any]], 
                  query_sets: List[Dict[str, Any]],
                  num_meta_steps: int = 100) -> Dict[str, float]:
        """
        Perform meta-training on multiple tasks
        
        Args:
            support_sets: List of support sets for each task
            query_sets: List of query sets for each task
            num_meta_steps: Number of meta-training steps
            
        Returns:
            Meta-training metrics
        """
        try:
            logger.info(f"Starting meta-training with {len(support_sets)} tasks")
            
            meta_losses = []
            adaptation_accuracies = []
            
            for step in range(num_meta_steps):
                # Sample a batch of tasks
                task_indices = np.random.choice(len(support_sets), size=min(4, len(support_sets)), replace=False)
                
                # Initialize task-specific parameters
                task_gradients = []
                task_losses = []
                
                for task_idx in task_indices:
                    support_set = support_sets[task_idx]
                    query_set = query_sets[task_idx]
                    
                    # Inner loop: adapt to this specific task
                    adapted_params, inner_loss = self._inner_loop_adaptation(
                        support_set, self.num_inner_steps
                    )
                    
                    # Evaluate on query set with adapted parameters
                    query_loss = self._evaluate_on_query_set(query_set, adapted_params)
                    
                    task_gradients.append(self._compute_task_gradient(adapted_params))
                    task_losses.append(query_loss)
                
                # Meta-update: update base model parameters
                meta_loss = self._meta_update(task_gradients, task_losses)
                meta_losses.append(meta_loss)
                
                # Calculate adaptation accuracy
                if step % 10 == 0:
                    accuracy = self._evaluate_adaptation_accuracy(support_sets, query_sets)
                    adaptation_accuracies.append(accuracy)
                    logger.debug(f"Meta-step {step}: meta_loss={meta_loss:.4f}, accuracy={accuracy:.4f}")
            
            metrics = {
                'final_meta_loss': meta_losses[-1] if meta_losses else 0.0,
                'avg_meta_loss': np.mean(meta_losses),
                'final_accuracy': adaptation_accuracies[-1] if adaptation_accuracies else 0.0,
                'avg_accuracy': np.mean(adaptation_accuracies),
                'meta_steps': num_meta_steps
            }
            
            logger.info(f"Meta-training completed: final_loss={metrics['final_meta_loss']:.4f}, final_accuracy={metrics['final_accuracy']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Meta-training failed: {e}")
            return {"error": str(e)}
    
    def few_shot_adaptation(self, 
                          support_patterns: List[Dict[str, Any]], 
                          support_labels: List[int],
                          adaptation_steps: int = None) -> Dict[str, Any]:
        """
        Perform few-shot adaptation to new attack patterns
        
        Args:
            support_patterns: Support set of attack patterns
            support_labels: Corresponding labels
            adaptation_steps: Number of adaptation steps (default: self.num_inner_steps)
            
        Returns:
            Adaptation results
        """
        try:
            if adaptation_steps is None:
                adaptation_steps = self.num_inner_steps
            
            # Create support set
            support_set = {
                'patterns': support_patterns,
                'labels': support_labels
            }
            
            # Perform inner loop adaptation
            adapted_params, adaptation_loss = self._inner_loop_adaptation(
                support_set, adaptation_steps
            )
            
            # Evaluate adaptation quality
            adaptation_accuracy = self._evaluate_adaptation_quality(support_set, adapted_params)
            
            # Store adaptation history
            adaptation_record = {
                'support_patterns': support_patterns,
                'support_labels': support_labels,
                'adapted_params': adapted_params,
                'adaptation_loss': adaptation_loss,
                'adaptation_accuracy': adaptation_accuracy,
                'timestamp': np.datetime64('now')
            }
            self.adaptation_history.append(adaptation_record)
            
            results = {
                'adaptation_success': True,
                'adaptation_loss': adaptation_loss,
                'adaptation_accuracy': adaptation_accuracy,
                'adapted_params': adapted_params,
                'adaptation_steps': adaptation_steps
            }
            
            logger.info(f"Few-shot adaptation completed: accuracy={adaptation_accuracy:.4f}, loss={adaptation_loss:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Few-shot adaptation failed: {e}")
            return {"error": str(e), "adaptation_success": False}
    
    def _inner_loop_adaptation(self, 
                             support_set: Dict[str, Any], 
                             num_steps: int) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Perform inner loop adaptation for a specific task
        
        Args:
            support_set: Support set for the task
            num_steps: Number of adaptation steps
            
        Returns:
            Tuple of (adapted_parameters, adaptation_loss)
        """
        try:
            # Initialize task-specific parameters as copy of base model
            adapted_params = {name: param.clone() for name, param in self.base_model.named_parameters()}
            
            # Inner loop optimization
            for step in range(num_steps):
                # Forward pass with current parameters
                loss = self._compute_task_loss(support_set, adapted_params)
                
                # Compute gradients
                gradients = self._compute_gradients(loss, adapted_params)
                
                # Update parameters using gradient descent
                for name, param in adapted_params.items():
                    if name in gradients:
                        adapted_params[name] = param - self.inner_lr * gradients[name]
            
            # Final loss computation
            final_loss = self._compute_task_loss(support_set, adapted_params)
            
            return adapted_params, final_loss.item()
            
        except Exception as e:
            logger.error(f"Inner loop adaptation failed: {e}")
            return {}, float('inf')
    
    def _compute_task_loss(self, 
                         support_set: Dict[str, Any], 
                         params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for a specific task with given parameters
        
        Args:
            support_set: Support set for the task
            params: Model parameters to use
            
        Returns:
            Task loss
        """
        try:
            patterns = support_set['patterns']
            labels = support_set['labels']
            
            if not patterns or not labels:
                return torch.tensor(0.0, device=self.device)
            
            # Convert patterns to model inputs
            inputs = self._prepare_model_inputs(patterns)
            targets = torch.tensor(labels, dtype=torch.long, device=self.device)
            
            # Forward pass with given parameters
            outputs = self._forward_with_params(inputs, params)
            
            # Compute loss
            loss = F.cross_entropy(outputs, targets)
            
            return loss
            
        except Exception as e:
            logger.error(f"Task loss computation failed: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def _prepare_model_inputs(self, patterns: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Prepare model inputs from attack patterns
        
        Args:
            patterns: List of attack patterns
            
        Returns:
            Prepared model inputs
        """
        try:
            # Extract text from patterns
            texts = []
            for pattern in patterns:
                if isinstance(pattern, dict):
                    text = pattern.get('text', pattern.get('pattern', ''))
                else:
                    text = str(pattern)
                texts.append(text)
            
            # Tokenize texts (simplified - in practice, use proper tokenizer)
            # This is a placeholder - replace with actual tokenization
            max_length = 512
            input_ids = torch.zeros(len(texts), max_length, dtype=torch.long, device=self.device)
            
            for i, text in enumerate(texts):
                # Simple character-based encoding (replace with proper tokenization)
                text_encoded = [ord(c) % 1000 for c in text[:max_length]]
                input_ids[i, :len(text_encoded)] = torch.tensor(text_encoded, device=self.device)
            
            return input_ids
            
        except Exception as e:
            logger.error(f"Model input preparation failed: {e}")
            return torch.zeros(1, 512, dtype=torch.long, device=self.device)
    
    def _forward_with_params(self, 
                           inputs: torch.Tensor, 
                           params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with specific parameters
        
        Args:
            inputs: Model inputs
            params: Parameters to use
            
        Returns:
            Model outputs
        """
        try:
            # Temporarily replace model parameters
            original_params = {}
            for name, param in self.base_model.named_parameters():
                original_params[name] = param.data.clone()
                param.data = params[name]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.base_model(inputs)
            
            # Restore original parameters
            for name, param in self.base_model.named_parameters():
                param.data = original_params[name]
            
            return outputs
            
        except Exception as e:
            logger.error(f"Forward pass with params failed: {e}")
            return torch.zeros(inputs.size(0), 2, device=self.device)
    
    def _compute_gradients(self, 
                         loss: torch.Tensor, 
                         params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute gradients with respect to parameters
        
        Args:
            loss: Loss tensor
            params: Parameters to compute gradients for
            
        Returns:
            Dictionary of gradients
        """
        try:
            # Create a temporary model with the given parameters
            temp_model = copy.deepcopy(self.base_model)
            for name, param in temp_model.named_parameters():
                param.data = params[name]
                param.requires_grad = True
            
            # Compute gradients
            gradients = torch.autograd.grad(loss, temp_model.parameters(), create_graph=True)
            
            # Convert to dictionary
            grad_dict = {}
            for (name, _), grad in zip(temp_model.named_parameters(), gradients):
                grad_dict[name] = grad
            
            return grad_dict
            
        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            return {}
    
    def _compute_task_gradient(self, 
                             adapted_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute task gradient for meta-update
        
        Args:
            adapted_params: Adapted parameters for the task
            
        Returns:
            Task gradient
        """
        try:
            # Compute gradient from adapted parameters to base parameters
            task_gradient = {}
            for name, adapted_param in adapted_params.items():
                if name in dict(self.base_model.named_parameters()):
                    base_param = dict(self.base_model.named_parameters())[name]
                    task_gradient[name] = adapted_param - base_param
            
            return task_gradient
            
        except Exception as e:
            logger.error(f"Task gradient computation failed: {e}")
            return {}
    
    def _meta_update(self, 
                    task_gradients: List[Dict[str, torch.Tensor]], 
                    task_losses: List[float]) -> float:
        """
        Perform meta-update using task gradients
        
        Args:
            task_gradients: List of task gradients
            task_losses: List of task losses
            
        Returns:
            Meta-loss
        """
        try:
            # Average task gradients
            avg_gradient = {}
            for name in task_gradients[0].keys():
                avg_gradient[name] = torch.stack([grad[name] for grad in task_gradients]).mean(0)
            
            # Update base model parameters
            with torch.no_grad():
                for name, param in self.base_model.named_parameters():
                    if name in avg_gradient:
                        param.data += self.meta_lr * avg_gradient[name]
            
            # Return average meta-loss
            return np.mean(task_losses)
            
        except Exception as e:
            logger.error(f"Meta-update failed: {e}")
            return 0.0
    
    def _evaluate_on_query_set(self, 
                             query_set: Dict[str, Any], 
                             adapted_params: Dict[str, torch.Tensor]) -> float:
        """
        Evaluate adapted model on query set
        
        Args:
            query_set: Query set for evaluation
            adapted_params: Adapted parameters
            
        Returns:
            Query loss
        """
        try:
            return self._compute_task_loss(query_set, adapted_params).item()
            
        except Exception as e:
            logger.error(f"Query set evaluation failed: {e}")
            return float('inf')
    
    def _evaluate_adaptation_accuracy(self, 
                                    support_sets: List[Dict[str, Any]], 
                                    query_sets: List[Dict[str, Any]]) -> float:
        """
        Evaluate adaptation accuracy across all tasks
        
        Args:
            support_sets: List of support sets
            query_sets: List of query sets
            
        Returns:
            Average adaptation accuracy
        """
        try:
            accuracies = []
            
            for support_set, query_set in zip(support_sets, query_sets):
                # Adapt to support set
                adapted_params, _ = self._inner_loop_adaptation(support_set, self.num_inner_steps)
                
                # Evaluate on query set
                query_patterns = query_set['patterns']
                query_labels = query_set['labels']
                
                if query_patterns and query_labels:
                    inputs = self._prepare_model_inputs(query_patterns)
                    outputs = self._forward_with_params(inputs, adapted_params)
                    predictions = torch.argmax(outputs, dim=-1)
                    
                    accuracy = (predictions == torch.tensor(query_labels, device=self.device)).float().mean().item()
                    accuracies.append(accuracy)
            
            return np.mean(accuracies) if accuracies else 0.0
            
        except Exception as e:
            logger.error(f"Adaptation accuracy evaluation failed: {e}")
            return 0.0
    
    def _evaluate_adaptation_quality(self, 
                                   support_set: Dict[str, Any], 
                                   adapted_params: Dict[str, torch.Tensor]) -> float:
        """
        Evaluate quality of adaptation
        
        Args:
            support_set: Support set used for adaptation
            adapted_params: Adapted parameters
            
        Returns:
            Adaptation quality score
        """
        try:
            # Compute loss on support set with adapted parameters
            support_loss = self._compute_task_loss(support_set, adapted_params).item()
            
            # Convert to accuracy (simplified)
            accuracy = max(0.0, 1.0 - support_loss)
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Adaptation quality evaluation failed: {e}")
            return 0.0
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get adaptation history"""
        return self.adaptation_history
    
    def clear_adaptation_history(self) -> None:
        """Clear adaptation history"""
        self.adaptation_history = []
    
    def get_attack_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about attack pattern memory"""
        try:
            stats = {}
            for category, patterns in self.attack_memory.items():
                stats[category] = {
                    'count': len(patterns),
                    'attack_ratio': sum(1 for p in patterns if p['label'] == 1) / len(patterns) if patterns else 0.0
                }
            
            return {
                'categories': list(self.attack_memory.keys()),
                'total_patterns': sum(len(patterns) for patterns in self.attack_memory.values()),
                'category_stats': stats
            }
            
        except Exception as e:
            logger.error(f"Attack memory stats computation failed: {e}")
            return {"error": str(e)}
