"""
Universal Adversarial Triggers
Implementation of Wallace et al., 2019 technique for universal adversarial triggers
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Union
from torch.optim import Adam
import json

logger = logging.getLogger(__name__)


class UniversalTriggerGenerator:
    """
    Universal Adversarial Triggers (Wallace et al., 2019)
    Generate triggers that work across multiple inputs
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize universal trigger generator
        
        Args:
            model: Target model to attack
            tokenizer: Tokenizer for the model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Get vocabulary info
        self.vocab_size = len(tokenizer.vocab)
        self.pad_token_id = tokenizer.pad_token_id
        self.unk_token_id = tokenizer.unk_token_id
        
        logger.info(f"✅ Initialized UniversalTriggerGenerator with vocab_size={self.vocab_size}")
    
    def generate_universal_trigger(self, 
                                 trigger_length: int = 5,
                                 num_iterations: int = 1000,
                                 dataset_samples: List[str] = None,
                                 learning_rate: float = 0.01,
                                 target_class: Optional[int] = None,
                                 targeted: bool = False,
                                 trigger_position: str = 'prepend') -> Tuple[str, Dict]:
        """
        Generate universal adversarial trigger
        
        Args:
            trigger_length: Length of trigger in tokens
            num_iterations: Number of optimization iterations
            dataset_samples: List of sample texts to optimize against
            learning_rate: Learning rate for optimization
            target_class: Target class for targeted attacks
            targeted: Whether to perform targeted attack
            trigger_position: Where to place trigger ('prepend', 'append', 'insert')
            
        Returns:
            Tuple of (trigger_text, generation_info)
        """
        try:
            # 1. Initialize random trigger
            trigger_tokens = torch.randint(0, self.vocab_size, (trigger_length,), device=self.device)
            trigger_tokens.requires_grad = True
            
            # 2. Prepare dataset samples
            if dataset_samples is None:
                dataset_samples = self._get_default_samples()
            
            # Convert samples to tokenized format
            sample_inputs = []
            for sample in dataset_samples:
                inputs = self.tokenizer(sample, return_tensors="pt", padding=True, truncation=True)
                sample_inputs.append({
                    'input_ids': inputs.input_ids.to(self.device),
                    'attention_mask': inputs.attention_mask.to(self.device)
                })
            
            # 3. Optimize trigger
            optimizer = Adam([trigger_tokens], lr=learning_rate)
            
            best_trigger = None
            best_success_rate = 0.0
            success_rates = []
            
            for iteration in range(num_iterations):
                optimizer.zero_grad()
                
                total_loss = 0.0
                successful_attacks = 0
                
                # Test trigger on all samples
                for sample_input in sample_inputs:
                    # Insert trigger into sample
                    attacked_input = self._insert_trigger(
                        sample_input, trigger_tokens, trigger_position
                    )
                    
                    # Forward pass
                    outputs = self.model(**attacked_input)
                    logits = outputs.logits
                    
                    # Calculate loss
                    if targeted and target_class is not None:
                        # Targeted: minimize loss for target class
                        loss = -F.cross_entropy(logits, torch.tensor([target_class]).to(self.device))
                    else:
                        # Untargeted: maximize loss for current prediction
                        current_pred = torch.argmax(logits, dim=-1)
                        loss = F.cross_entropy(logits, current_pred)
                    
                    total_loss += loss
                    
                    # Check if attack succeeded
                    if self._is_attack_successful(sample_input, attacked_input, target_class, targeted):
                        successful_attacks += 1
                
                # Average loss
                avg_loss = total_loss / len(sample_inputs)
                success_rate = successful_attacks / len(sample_inputs)
                success_rates.append(success_rate)
                
                # Backward pass
                avg_loss.backward()
                optimizer.step()
                
                # Project to valid token IDs
                with torch.no_grad():
                    trigger_tokens = torch.clamp(trigger_tokens, 0, self.vocab_size - 1)
                    trigger_tokens = trigger_tokens.round().long()
                
                # Track best trigger
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_trigger = trigger_tokens.clone()
                
                # Log progress
                if iteration % 100 == 0:
                    logger.debug(f"Universal trigger iteration {iteration}: success_rate={success_rate:.3f}, loss={avg_loss.item():.4f}")
                
                # Early stopping if perfect success
                if success_rate >= 1.0:
                    break
            
            # 4. Convert best trigger to text
            if best_trigger is not None:
                trigger_text = self.tokenizer.decode(best_trigger, skip_special_tokens=True)
            else:
                trigger_text = self.tokenizer.decode(trigger_tokens, skip_special_tokens=True)
            
            # 5. Generate info
            generation_info = {
                "method": "UniversalTrigger",
                "trigger_length": trigger_length,
                "num_iterations": num_iterations,
                "learning_rate": learning_rate,
                "targeted": targeted,
                "target_class": target_class,
                "trigger_position": trigger_position,
                "final_success_rate": best_success_rate,
                "success_rates": success_rates,
                "trigger_text": trigger_text,
                "trigger_tokens": best_trigger.tolist() if best_trigger is not None else trigger_tokens.tolist()
            }
            
            logger.info(f"Universal trigger generated: success_rate={best_success_rate:.3f}, trigger='{trigger_text}'")
            return trigger_text, generation_info
            
        except Exception as e:
            logger.error(f"Universal trigger generation failed: {e}")
            return "", {"error": str(e), "method": "UniversalTrigger"}
    
    def test_trigger(self, trigger_text: str, test_samples: List[str],
                    trigger_position: str = 'prepend') -> Dict:
        """
        Test universal trigger on new samples
        
        Args:
            trigger_text: Generated trigger text
            test_samples: List of test samples
            trigger_position: Where to place trigger
            
        Returns:
            Test results dictionary
        """
        try:
            # Tokenize trigger
            trigger_tokens = self.tokenizer.encode(trigger_text, add_special_tokens=False)
            trigger_tensor = torch.tensor(trigger_tokens, device=self.device)
            
            results = {
                "trigger_text": trigger_text,
                "trigger_position": trigger_position,
                "test_samples": len(test_samples),
                "successful_attacks": 0,
                "failed_attacks": 0,
                "success_rate": 0.0,
                "sample_results": []
            }
            
            for i, sample in enumerate(test_samples):
                # Get original prediction
                orig_inputs = self.tokenizer(sample, return_tensors="pt", padding=True, truncation=True)
                orig_inputs = {k: v.to(self.device) for k, v in orig_inputs.items()}
                
                with torch.no_grad():
                    orig_outputs = self.model(**orig_inputs)
                    orig_pred = torch.argmax(orig_outputs.logits, dim=-1).item()
                    orig_confidence = F.softmax(orig_outputs.logits, dim=-1).max().item()
                
                # Apply trigger
                attacked_input = self._insert_trigger_tokens(orig_inputs, trigger_tensor, trigger_position)
                
                with torch.no_grad():
                    adv_outputs = self.model(**attacked_input)
                    adv_pred = torch.argmax(adv_outputs.logits, dim=-1).item()
                    adv_confidence = F.softmax(adv_outputs.logits, dim=-1).max().item()
                
                # Check if attack succeeded
                attack_success = orig_pred != adv_pred
                
                if attack_success:
                    results["successful_attacks"] += 1
                else:
                    results["failed_attacks"] += 1
                
                results["sample_results"].append({
                    "sample": sample,
                    "original_prediction": orig_pred,
                    "adversarial_prediction": adv_pred,
                    "original_confidence": orig_confidence,
                    "adversarial_confidence": adv_confidence,
                    "attack_success": attack_success
                })
            
            results["success_rate"] = results["successful_attacks"] / len(test_samples)
            
            logger.info(f"Trigger test completed: success_rate={results['success_rate']:.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Trigger testing failed: {e}")
            return {"error": str(e)}
    
    def _insert_trigger(self, sample_input: Dict, trigger_tokens: torch.Tensor, 
                       position: str) -> Dict:
        """Insert trigger into sample input"""
        input_ids = sample_input['input_ids'].clone()
        attention_mask = sample_input['attention_mask'].clone()
        
        if position == 'prepend':
            # Prepend trigger to input
            new_input_ids = torch.cat([trigger_tokens.unsqueeze(0), input_ids], dim=1)
            new_attention_mask = torch.cat([
                torch.ones(1, len(trigger_tokens), device=self.device),
                attention_mask
            ], dim=1)
        elif position == 'append':
            # Append trigger to input
            new_input_ids = torch.cat([input_ids, trigger_tokens.unsqueeze(0)], dim=1)
            new_attention_mask = torch.cat([
                attention_mask,
                torch.ones(1, len(trigger_tokens), device=self.device)
            ], dim=1)
        else:
            # Insert in middle (simplified)
            mid_point = input_ids.shape[1] // 2
            new_input_ids = torch.cat([
                input_ids[:, :mid_point],
                trigger_tokens.unsqueeze(0),
                input_ids[:, mid_point:]
            ], dim=1)
            new_attention_mask = torch.cat([
                attention_mask[:, :mid_point],
                torch.ones(1, len(trigger_tokens), device=self.device),
                attention_mask[:, mid_point:]
            ], dim=1)
        
        return {
            'input_ids': new_input_ids,
            'attention_mask': new_attention_mask
        }
    
    def _insert_trigger_tokens(self, sample_input: Dict, trigger_tokens: torch.Tensor,
                              position: str) -> Dict:
        """Insert trigger tokens into sample input (for testing)"""
        return self._insert_trigger(sample_input, trigger_tokens, position)
    
    def _is_attack_successful(self, original_input: Dict, attacked_input: Dict,
                            target_class: Optional[int], targeted: bool) -> bool:
        """Check if attack was successful"""
        try:
            with torch.no_grad():
                # Get original prediction
                orig_outputs = self.model(**original_input)
                orig_pred = torch.argmax(orig_outputs.logits, dim=-1).item()
                
                # Get adversarial prediction
                adv_outputs = self.model(**attacked_input)
                adv_pred = torch.argmax(adv_outputs.logits, dim=-1).item()
                
                if targeted and target_class is not None:
                    return adv_pred == target_class
                else:
                    return orig_pred != adv_pred
                    
        except Exception as e:
            logger.error(f"Attack success check failed: {e}")
            return False
    
    def _get_default_samples(self) -> List[str]:
        """Get default sample texts for trigger optimization"""
        return [
            "This is a normal text sample.",
            "The weather is nice today.",
            "I like to read books.",
            "Machine learning is interesting.",
            "The cat sat on the mat.",
            "Hello world, how are you?",
            "This is a test message.",
            "I enjoy programming.",
            "The sun is shining bright.",
            "Technology advances quickly."
        ]
    
    def generate_multiple_triggers(self, 
                                 num_triggers: int = 5,
                                 trigger_length: int = 5,
                                 num_iterations: int = 500,
                                 dataset_samples: List[str] = None) -> List[Tuple[str, Dict]]:
        """
        Generate multiple universal triggers with different random seeds
        
        Args:
            num_triggers: Number of triggers to generate
            trigger_length: Length of each trigger
            num_iterations: Iterations per trigger
            dataset_samples: Sample texts for optimization
            
        Returns:
            List of (trigger_text, generation_info) tuples
        """
        triggers = []
        
        for i in range(num_triggers):
            # Set random seed for diversity
            torch.manual_seed(i * 42)
            np.random.seed(i * 42)
            random.seed(i * 42)
            
            logger.info(f"Generating trigger {i+1}/{num_triggers}")
            
            trigger_text, info = self.generate_universal_trigger(
                trigger_length=trigger_length,
                num_iterations=num_iterations,
                dataset_samples=dataset_samples
            )
            
            triggers.append((trigger_text, info))
        
        return triggers
    
    def analyze_trigger_effectiveness(self, triggers: List[Tuple[str, Dict]], 
                                    test_samples: List[str]) -> Dict:
        """
        Analyze effectiveness of multiple triggers
        
        Args:
            triggers: List of generated triggers
            test_samples: Test samples to evaluate on
            
        Returns:
            Analysis results
        """
        analysis = {
            "num_triggers": len(triggers),
            "test_samples": len(test_samples),
            "trigger_performance": [],
            "best_trigger": None,
            "average_success_rate": 0.0
        }
        
        best_success_rate = 0.0
        
        for i, (trigger_text, generation_info) in enumerate(triggers):
            # Test trigger
            test_results = self.test_trigger(trigger_text, test_samples)
            
            performance = {
                "trigger_index": i,
                "trigger_text": trigger_text,
                "generation_success_rate": generation_info.get("final_success_rate", 0.0),
                "test_success_rate": test_results.get("success_rate", 0.0),
                "test_results": test_results
            }
            
            analysis["trigger_performance"].append(performance)
            
            # Track best trigger
            if test_results.get("success_rate", 0.0) > best_success_rate:
                best_success_rate = test_results.get("success_rate", 0.0)
                analysis["best_trigger"] = performance
        
        # Calculate average success rate
        success_rates = [p["test_success_rate"] for p in analysis["trigger_performance"]]
        analysis["average_success_rate"] = sum(success_rates) / len(success_rates) if success_rates else 0.0
        
        logger.info(f"Trigger analysis completed: avg_success_rate={analysis['average_success_rate']:.3f}")
        return analysis
