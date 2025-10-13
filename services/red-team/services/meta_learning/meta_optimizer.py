"""
Meta-Optimizer for Attack Adaptation
Advanced optimization techniques for meta-learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


class MetaOptimizer:
    """
    Advanced meta-optimizer for attack adaptation
    Implements various meta-learning optimization strategies
    """
    
    def __init__(self, 
                 model: nn.Module,
                 meta_lr: float = 0.001,
                 inner_lr: float = 0.01,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize meta-optimizer
        
        Args:
            model: Model to optimize
            meta_lr: Meta-learning rate
            inner_lr: Inner loop learning rate
            device: Device to run on
        """
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.device = device
        
        # Initialize optimizers
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        self.inner_optimizer = optim.SGD(model.parameters(), lr=inner_lr)
        
        # Meta-learning state
        self.meta_gradients = []
        self.task_losses = []
        self.adaptation_history = []
        
        logger.info(f"✅ Meta-Optimizer initialized: meta_lr={meta_lr}, inner_lr={inner_lr}")
    
    def reptile_update(self, 
                     task_batch: List[Dict[str, Any]], 
                     num_inner_steps: int = 5) -> Dict[str, float]:
        """
        Perform Reptile meta-learning update
        
        Args:
            task_batch: Batch of tasks for meta-update
            num_inner_steps: Number of inner loop steps
            
        Returns:
            Update metrics
        """
        try:
            logger.info(f"Starting Reptile update with {len(task_batch)} tasks")
            
            # Store original parameters
            original_params = {name: param.clone() for name, param in self.model.named_parameters()}
            
            # Perform inner loop updates for each task
            task_updates = []
            task_losses = []
            
            for task in task_batch:
                # Inner loop update
                task_params, task_loss = self._inner_loop_update(task, num_inner_steps)
                task_updates.append(task_params)
                task_losses.append(task_loss)
            
            # Compute Reptile update
            reptile_update = self._compute_reptile_update(original_params, task_updates)
            
            # Apply meta-update
            self._apply_meta_update(reptile_update)
            
            metrics = {
                'method': 'reptile',
                'num_tasks': len(task_batch),
                'avg_task_loss': np.mean(task_losses),
                'update_norm': torch.norm(torch.stack([torch.norm(update) for update in reptile_update.values()])).item()
            }
            
            logger.info(f"Reptile update completed: avg_loss={metrics['avg_task_loss']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Reptile update failed: {e}")
            return {"error": str(e)}
    
    def maml_update(self, 
                   task_batch: List[Dict[str, Any]], 
                   num_inner_steps: int = 5) -> Dict[str, float]:
        """
        Perform MAML meta-learning update
        
        Args:
            task_batch: Batch of tasks for meta-update
            num_inner_steps: Number of inner loop steps
            
        Returns:
            Update metrics
        """
        try:
            logger.info(f"Starting MAML update with {len(task_batch)} tasks")
            
            # Store original parameters
            original_params = {name: param.clone() for name, param in self.model.named_parameters()}
            
            # Compute task gradients
            task_gradients = []
            task_losses = []
            
            for task in task_batch:
                # Inner loop update
                adapted_params, inner_loss = self._inner_loop_update(task, num_inner_steps)
                
                # Compute task gradient
                task_gradient = self._compute_task_gradient(original_params, adapted_params)
                task_gradients.append(task_gradient)
                task_losses.append(inner_loss)
            
            # Compute meta-gradient
            meta_gradient = self._compute_meta_gradient(task_gradients)
            
            # Apply meta-update
            self._apply_meta_gradient(meta_gradient)
            
            metrics = {
                'method': 'maml',
                'num_tasks': len(task_batch),
                'avg_task_loss': np.mean(task_losses),
                'meta_gradient_norm': torch.norm(torch.stack([torch.norm(grad) for grad in meta_gradient.values()])).item()
            }
            
            logger.info(f"MAML update completed: avg_loss={metrics['avg_task_loss']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"MAML update failed: {e}")
            return {"error": str(e)}
    
    def first_order_maml_update(self, 
                              task_batch: List[Dict[str, Any]], 
                              num_inner_steps: int = 5) -> Dict[str, float]:
        """
        Perform first-order MAML update (simplified version)
        
        Args:
            task_batch: Batch of tasks for meta-update
            num_inner_steps: Number of inner loop steps
            
        Returns:
            Update metrics
        """
        try:
            logger.info(f"Starting First-Order MAML update with {len(task_batch)} tasks")
            
            # Store original parameters
            original_params = {name: param.clone() for name, param in self.model.named_parameters()}
            
            # Compute task updates
            task_updates = []
            task_losses = []
            
            for task in task_batch:
                # Single inner loop step
                task_params, task_loss = self._inner_loop_update(task, 1)
                task_updates.append(task_params)
                task_losses.append(task_loss)
            
            # Average task updates
            avg_update = self._average_parameter_updates(original_params, task_updates)
            
            # Apply meta-update
            self._apply_meta_update(avg_update)
            
            metrics = {
                'method': 'first_order_maml',
                'num_tasks': len(task_batch),
                'avg_task_loss': np.mean(task_losses),
                'update_norm': torch.norm(torch.stack([torch.norm(update) for update in avg_update.values()])).item()
            }
            
            logger.info(f"First-Order MAML update completed: avg_loss={metrics['avg_task_loss']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"First-Order MAML update failed: {e}")
            return {"error": str(e)}
    
    def _inner_loop_update(self, 
                          task: Dict[str, Any], 
                          num_steps: int) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Perform inner loop update for a single task
        
        Args:
            task: Task data
            num_steps: Number of update steps
            
        Returns:
            Tuple of (updated_parameters, final_loss)
        """
        try:
            # Create a copy of the model for this task
            task_model = copy.deepcopy(self.model)
            task_optimizer = optim.SGD(task_model.parameters(), lr=self.inner_lr)
            
            # Inner loop optimization
            for step in range(num_steps):
                task_optimizer.zero_grad()
                
                # Compute task loss
                loss = self._compute_task_loss(task, task_model)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                task_optimizer.step()
            
            # Get final parameters
            final_params = {name: param.clone() for name, param in task_model.named_parameters()}
            final_loss = loss.item()
            
            return final_params, final_loss
            
        except Exception as e:
            logger.error(f"Inner loop update failed: {e}")
            return {}, float('inf')
    
    def _compute_task_loss(self, 
                          task: Dict[str, Any], 
                          model: nn.Module) -> torch.Tensor:
        """
        Compute loss for a specific task
        
        Args:
            task: Task data
            model: Model to use
            
        Returns:
            Task loss
        """
        try:
            # Extract task data
            support_data = task.get('support_data', {})
            support_labels = task.get('support_labels', [])
            
            if not support_data or not support_labels:
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Prepare inputs
            inputs = self._prepare_task_inputs(support_data)
            targets = torch.tensor(support_labels, dtype=torch.long, device=self.device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = F.cross_entropy(outputs, targets)
            
            return loss
            
        except Exception as e:
            logger.error(f"Task loss computation failed: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def _prepare_task_inputs(self, task_data: Dict[str, Any]) -> torch.Tensor:
        """
        Prepare model inputs from task data
        
        Args:
            task_data: Task data dictionary
            
        Returns:
            Prepared inputs
        """
        try:
            # Extract patterns from task data
            patterns = task_data.get('patterns', [])
            
            if not patterns:
                return torch.zeros(1, 512, dtype=torch.long, device=self.device)
            
            # Convert patterns to model inputs
            max_length = 512
            batch_size = len(patterns)
            inputs = torch.zeros(batch_size, max_length, dtype=torch.long, device=self.device)
            
            for i, pattern in enumerate(patterns):
                if isinstance(pattern, dict):
                    text = pattern.get('text', pattern.get('pattern', ''))
                else:
                    text = str(pattern)
                
                # Simple tokenization (replace with proper tokenizer)
                text_encoded = [ord(c) % 1000 for c in text[:max_length]]
                inputs[i, :len(text_encoded)] = torch.tensor(text_encoded, device=self.device)
            
            return inputs
            
        except Exception as e:
            logger.error(f"Task input preparation failed: {e}")
            return torch.zeros(1, 512, dtype=torch.long, device=self.device)
    
    def _compute_reptile_update(self, 
                              original_params: Dict[str, torch.Tensor], 
                              task_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Compute Reptile update
        
        Args:
            original_params: Original model parameters
            task_updates: List of task-specific parameter updates
            
        Returns:
            Reptile update
        """
        try:
            # Average the task updates
            avg_update = {}
            for name in original_params.keys():
                updates = [task_update[name] for task_update in task_updates if name in task_update]
                if updates:
                    avg_update[name] = torch.stack(updates).mean(0)
                else:
                    avg_update[name] = torch.zeros_like(original_params[name])
            
            return avg_update
            
        except Exception as e:
            logger.error(f"Reptile update computation failed: {e}")
            return {}
    
    def _compute_task_gradient(self, 
                             original_params: Dict[str, torch.Tensor], 
                             adapted_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute task gradient for MAML
        
        Args:
            original_params: Original parameters
            adapted_params: Adapted parameters
            
        Returns:
            Task gradient
        """
        try:
            task_gradient = {}
            for name in original_params.keys():
                if name in adapted_params:
                    task_gradient[name] = adapted_params[name] - original_params[name]
                else:
                    task_gradient[name] = torch.zeros_like(original_params[name])
            
            return task_gradient
            
        except Exception as e:
            logger.error(f"Task gradient computation failed: {e}")
            return {}
    
    def _compute_meta_gradient(self, 
                             task_gradients: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Compute meta-gradient from task gradients
        
        Args:
            task_gradients: List of task gradients
            
        Returns:
            Meta-gradient
        """
        try:
            meta_gradient = {}
            for name in task_gradients[0].keys():
                gradients = [task_grad[name] for task_grad in task_gradients if name in task_grad]
                if gradients:
                    meta_gradient[name] = torch.stack(gradients).mean(0)
                else:
                    meta_gradient[name] = torch.zeros_like(task_gradients[0][name])
            
            return meta_gradient
            
        except Exception as e:
            logger.error(f"Meta-gradient computation failed: {e}")
            return {}
    
    def _average_parameter_updates(self, 
                                 original_params: Dict[str, torch.Tensor], 
                                 task_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Average parameter updates across tasks
        
        Args:
            original_params: Original parameters
            task_updates: List of task updates
            
        Returns:
            Averaged update
        """
        try:
            avg_update = {}
            for name in original_params.keys():
                updates = [task_update[name] for task_update in task_updates if name in task_update]
                if updates:
                    avg_update[name] = torch.stack(updates).mean(0)
                else:
                    avg_update[name] = torch.zeros_like(original_params[name])
            
            return avg_update
            
        except Exception as e:
            logger.error(f"Parameter update averaging failed: {e}")
            return {}
    
    def _apply_meta_update(self, update: Dict[str, torch.Tensor]) -> None:
        """
        Apply meta-update to model parameters
        
        Args:
            update: Parameter update to apply
        """
        try:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in update:
                        param.data += self.meta_lr * update[name]
            
        except Exception as e:
            logger.error(f"Meta-update application failed: {e}")
    
    def _apply_meta_gradient(self, meta_gradient: Dict[str, torch.Tensor]) -> None:
        """
        Apply meta-gradient using optimizer
        
        Args:
            meta_gradient: Meta-gradient to apply
        """
        try:
            # Zero gradients
            self.meta_optimizer.zero_grad()
            
            # Set gradients
            for name, param in self.model.named_parameters():
                if name in meta_gradient:
                    param.grad = meta_gradient[name]
            
            # Update parameters
            self.meta_optimizer.step()
            
        except Exception as e:
            logger.error(f"Meta-gradient application failed: {e}")
    
    def adaptive_meta_learning(self, 
                             task_batch: List[Dict[str, Any]], 
                             adaptation_strategy: str = 'auto') -> Dict[str, Any]:
        """
        Perform adaptive meta-learning with strategy selection
        
        Args:
            task_batch: Batch of tasks
            adaptation_strategy: Strategy to use ('auto', 'reptile', 'maml', 'first_order')
            
        Returns:
            Adaptation results
        """
        try:
            if adaptation_strategy == 'auto':
                # Automatically select strategy based on task characteristics
                strategy = self._select_adaptation_strategy(task_batch)
            else:
                strategy = adaptation_strategy
            
            logger.info(f"Using adaptation strategy: {strategy}")
            
            # Perform meta-learning update
            if strategy == 'reptile':
                results = self.reptile_update(task_batch)
            elif strategy == 'maml':
                results = self.maml_update(task_batch)
            elif strategy == 'first_order':
                results = self.first_order_maml_update(task_batch)
            else:
                results = {"error": f"Unknown strategy: {strategy}"}
            
            # Add strategy information
            results['strategy_used'] = strategy
            results['num_tasks'] = len(task_batch)
            
            return results
            
        except Exception as e:
            logger.error(f"Adaptive meta-learning failed: {e}")
            return {"error": str(e)}
    
    def _select_adaptation_strategy(self, task_batch: List[Dict[str, Any]]) -> str:
        """
        Automatically select adaptation strategy based on task characteristics
        
        Args:
            task_batch: Batch of tasks
            
        Returns:
            Selected strategy
        """
        try:
            # Analyze task characteristics
            num_tasks = len(task_batch)
            avg_task_size = np.mean([len(task.get('support_data', {}).get('patterns', [])) for task in task_batch])
            
            # Strategy selection logic
            if num_tasks < 3:
                return 'first_order'  # Few tasks, use simple method
            elif avg_task_size < 10:
                return 'reptile'  # Small tasks, use Reptile
            else:
                return 'maml'  # Large tasks, use full MAML
            
        except Exception as e:
            logger.error(f"Strategy selection failed: {e}")
            return 'first_order'  # Default fallback
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get optimization statistics
        
        Returns:
            Optimization statistics
        """
        try:
            stats = {
                'meta_lr': self.meta_lr,
                'inner_lr': self.inner_lr,
                'num_meta_gradients': len(self.meta_gradients),
                'num_task_losses': len(self.task_losses),
                'adaptation_history_length': len(self.adaptation_history),
                'device': str(self.device)
            }
            
            if self.task_losses:
                stats['avg_task_loss'] = np.mean(self.task_losses)
                stats['min_task_loss'] = np.min(self.task_losses)
                stats['max_task_loss'] = np.max(self.task_losses)
            
            return stats
            
        except Exception as e:
            logger.error(f"Optimization stats computation failed: {e}")
            return {"error": str(e)}
    
    def clear_history(self) -> None:
        """Clear optimization history"""
        try:
            self.meta_gradients.clear()
            self.task_losses.clear()
            self.adaptation_history.clear()
            
            logger.info("Cleared optimization history")
            
        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
