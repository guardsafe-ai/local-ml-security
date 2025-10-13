"""
Gradient-Based Adversarial Attacks
Implementation of FGSM, PGD, and C&W attacks for text models
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from torch.optim import Adam
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class GradientBasedAttacks:
    """
    State-of-the-art gradient-based adversarial attacks for text models
    Implements FGSM, PGD, and C&W attacks with research-grade algorithms
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize gradient-based attack generator
        
        Args:
            model: PyTorch model to attack
            tokenizer: Tokenizer for the model
            device: Device to run attacks on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Get embedding layer
        if hasattr(model, 'get_input_embeddings'):
            self.embedding_layer = model.get_input_embeddings()
        else:
            # For custom models, try to find embedding layer
            for name, module in model.named_modules():
                if 'embedding' in name.lower():
                    self.embedding_layer = module
                    break
            else:
                raise ValueError("Could not find embedding layer in model")
        
        self.vocab_size = self.embedding_layer.num_embeddings
        self.embedding_dim = self.embedding_layer.embedding_dim
        
        logger.info(f"✅ Initialized GradientBasedAttacks with vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}")
    
    def fgsm_attack(self, text: str, epsilon: float = 0.01, 
                   target_class: Optional[int] = None, 
                   targeted: bool = False) -> Tuple[str, float, Dict]:
        """
        Fast Gradient Sign Method (Goodfellow et al., 2014)
        
        Args:
            text: Input text to attack
            epsilon: Perturbation magnitude
            target_class: Target class for targeted attacks
            targeted: Whether to perform targeted attack
            
        Returns:
            Tuple of (adversarial_text, perturbation_norm, attack_info)
        """
        try:
            # 1. Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # 2. Get embeddings with gradients enabled
            embeddings = self.embedding_layer(input_ids)
            embeddings.requires_grad = True
            
            # 3. Forward pass
            outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
            
            # 4. Calculate loss
            if targeted and target_class is not None:
                # Targeted: minimize loss for target class
                loss = -F.cross_entropy(outputs.logits, torch.tensor([target_class]).to(self.device))
            else:
                # Untargeted: maximize loss for current prediction
                current_pred = torch.argmax(outputs.logits, dim=-1)
                loss = F.cross_entropy(outputs.logits, current_pred)
            
            # 5. Backward pass to get gradients
            loss.backward()
            
            # 6. Generate adversarial perturbation
            perturbation = epsilon * embeddings.grad.sign()
            adversarial_embeddings = embeddings + perturbation
            
            # 7. Project back to nearest valid tokens
            adversarial_text, projection_info = self._project_to_vocab(adversarial_embeddings, input_ids)
            
            # 8. Calculate perturbation norm
            perturbation_norm = perturbation.norm().item()
            
            attack_info = {
                "method": "FGSM",
                "epsilon": epsilon,
                "targeted": targeted,
                "target_class": target_class,
                "perturbation_norm": perturbation_norm,
                "original_prediction": torch.argmax(outputs.logits, dim=-1).item(),
                "projection_info": projection_info
            }
            
            logger.debug(f"FGSM attack completed: norm={perturbation_norm:.4f}")
            return adversarial_text, perturbation_norm, attack_info
            
        except Exception as e:
            logger.error(f"FGSM attack failed: {e}")
            return text, 0.0, {"error": str(e)}
    
    def pgd_attack(self, text: str, epsilon: float = 0.1, alpha: float = 0.01, 
                  num_steps: int = 20, target_class: Optional[int] = None,
                  targeted: bool = False, norm: str = 'l2') -> Tuple[str, float, Dict]:
        """
        Projected Gradient Descent (Madry et al., 2017)
        Multi-step iterative attack stronger than FGSM
        
        Args:
            text: Input text to attack
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            num_steps: Number of PGD steps
            target_class: Target class for targeted attacks
            targeted: Whether to perform targeted attack
            norm: Norm constraint ('l2' or 'linf')
            
        Returns:
            Tuple of (adversarial_text, perturbation_norm, attack_info)
        """
        try:
            # 1. Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # 2. Get original embeddings
            original_embeddings = self.embedding_layer(input_ids)
            adversarial_embeddings = original_embeddings.clone()
            
            # 3. PGD iterations
            for step in range(num_steps):
                adversarial_embeddings.requires_grad = True
                
                # Forward pass
                outputs = self.model(inputs_embeds=adversarial_embeddings, attention_mask=attention_mask)
                
                # Calculate loss
                if targeted and target_class is not None:
                    loss = -F.cross_entropy(outputs.logits, torch.tensor([target_class]).to(self.device))
                else:
                    current_pred = torch.argmax(outputs.logits, dim=-1)
                    loss = F.cross_entropy(outputs.logits, current_pred)
                
                # Backward pass
                loss.backward()
                
                # Update embeddings with gradient
                with torch.no_grad():
                    if norm == 'l2':
                        # L2 norm constraint
                        grad_norm = adversarial_embeddings.grad.norm(p=2)
                        if grad_norm > 0:
                            adversarial_embeddings = adversarial_embeddings + alpha * adversarial_embeddings.grad / grad_norm
                    else:  # linf
                        # L-infinity norm constraint
                        adversarial_embeddings = adversarial_embeddings + alpha * adversarial_embeddings.grad.sign()
                    
                    # Project to epsilon ball
                    perturbation = adversarial_embeddings - original_embeddings
                    if norm == 'l2':
                        perturbation_norm = perturbation.norm(p=2)
                        if perturbation_norm > epsilon:
                            perturbation = perturbation * epsilon / perturbation_norm
                    else:  # linf
                        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
                    
                    adversarial_embeddings = (original_embeddings + perturbation).detach()
            
            # 4. Project back to valid tokens
            adversarial_text, projection_info = self._project_to_vocab(adversarial_embeddings, input_ids)
            
            # 5. Calculate final perturbation norm
            final_perturbation = adversarial_embeddings - original_embeddings
            perturbation_norm = final_perturbation.norm().item()
            
            attack_info = {
                "method": "PGD",
                "epsilon": epsilon,
                "alpha": alpha,
                "num_steps": num_steps,
                "targeted": targeted,
                "target_class": target_class,
                "norm": norm,
                "perturbation_norm": perturbation_norm,
                "projection_info": projection_info
            }
            
            logger.debug(f"PGD attack completed: norm={perturbation_norm:.4f}, steps={num_steps}")
            return adversarial_text, perturbation_norm, attack_info
            
        except Exception as e:
            logger.error(f"PGD attack failed: {e}")
            return text, 0.0, {"error": str(e)}
    
    def carlini_wagner_attack(self, text: str, confidence: float = 0, 
                            max_iterations: int = 1000, learning_rate: float = 0.01,
                            target_class: Optional[int] = None, 
                            targeted: bool = False) -> Tuple[str, float, Dict]:
        """
        Carlini & Wagner L2 attack (C&W) - Optimization-based
        Minimizes perturbation while ensuring misclassification
        
        Args:
            text: Input text to attack
            confidence: Confidence margin for attack success
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization
            target_class: Target class for targeted attacks
            targeted: Whether to perform targeted attack
            
        Returns:
            Tuple of (adversarial_text, perturbation_norm, attack_info)
        """
        try:
            # 1. Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # 2. Get original embeddings
            original_embeddings = self.embedding_layer(input_ids)
            
            # 3. Initialize perturbation variables
            perturbation = torch.zeros_like(original_embeddings, requires_grad=True)
            optimizer = Adam([perturbation], lr=learning_rate)
            
            # 4. C&W optimization loop
            best_perturbation = None
            best_norm = float('inf')
            
            for iteration in range(max_iterations):
                optimizer.zero_grad()
                
                # Current adversarial embeddings
                adversarial_embeddings = original_embeddings + perturbation
                
                # Forward pass
                outputs = self.model(inputs_embeds=adversarial_embeddings, attention_mask=attention_mask)
                logits = outputs.logits
                
                # C&W loss function
                if targeted and target_class is not None:
                    # Targeted: minimize target class logit, maximize others
                    target_logit = logits[0, target_class]
                    other_logits = torch.cat([logits[0, :target_class], logits[0, target_class+1:]])
                    max_other_logit = torch.max(other_logits)
                    cw_loss = torch.clamp(max_other_logit - target_logit + confidence, min=0)
                else:
                    # Untargeted: minimize correct class logit, maximize others
                    correct_class = torch.argmax(logits, dim=-1)
                    correct_logit = logits[0, correct_class]
                    other_logits = torch.cat([logits[0, :correct_class], logits[0, correct_class+1:]])
                    max_other_logit = torch.max(other_logits)
                    cw_loss = torch.clamp(correct_logit - max_other_logit + confidence, min=0)
                
                # Total loss: perturbation norm + C&W loss
                perturbation_norm = torch.norm(perturbation, p=2)
                total_loss = perturbation_norm + 1000 * cw_loss  # High weight on C&W loss
                
                total_loss.backward()
                optimizer.step()
                
                # Check if attack succeeded
                if cw_loss.item() <= 0:  # Attack succeeded
                    current_norm = perturbation_norm.item()
                    if current_norm < best_norm:
                        best_norm = current_norm
                        best_perturbation = perturbation.clone().detach()
            
            # 5. Use best perturbation found
            if best_perturbation is not None:
                final_perturbation = best_perturbation
            else:
                final_perturbation = perturbation.detach()
            
            # 6. Project to valid tokens
            adversarial_embeddings = original_embeddings + final_perturbation
            adversarial_text, projection_info = self._project_to_vocab(adversarial_embeddings, input_ids)
            
            # 7. Calculate final metrics
            perturbation_norm = final_perturbation.norm().item()
            
            attack_info = {
                "method": "CarliniWagner",
                "confidence": confidence,
                "max_iterations": max_iterations,
                "targeted": targeted,
                "target_class": target_class,
                "perturbation_norm": perturbation_norm,
                "attack_success": best_perturbation is not None,
                "projection_info": projection_info
            }
            
            logger.debug(f"C&W attack completed: norm={perturbation_norm:.4f}, success={best_perturbation is not None}")
            return adversarial_text, perturbation_norm, attack_info
            
        except Exception as e:
            logger.error(f"C&W attack failed: {e}")
            return text, 0.0, {"error": str(e)}
    
    def _project_to_vocab(self, embeddings: torch.Tensor, original_ids: torch.Tensor) -> Tuple[str, Dict]:
        """
        Project continuous embeddings back to discrete vocabulary tokens
        
        Args:
            embeddings: Continuous embedding vectors
            original_ids: Original token IDs for reference
            
        Returns:
            Tuple of (adversarial_text, projection_info)
        """
        try:
            # Find nearest vocabulary embeddings
            embedding_weights = self.embedding_layer.weight  # [vocab_size, embedding_dim]
            
            # Calculate distances to all vocabulary embeddings
            distances = torch.cdist(embeddings, embedding_weights, p=2)  # [seq_len, vocab_size]
            
            # Find nearest tokens
            nearest_token_ids = torch.argmin(distances, dim=-1)  # [seq_len]
            
            # Convert to text
            adversarial_text = self.tokenizer.decode(nearest_token_ids, skip_special_tokens=True)
            
            # Calculate projection statistics
            original_tokens = original_ids[0].tolist()
            adversarial_tokens = nearest_token_ids.tolist()
            
            changed_tokens = sum(1 for orig, adv in zip(original_tokens, adversarial_tokens) if orig != adv)
            total_tokens = len(original_tokens)
            
            projection_info = {
                "changed_tokens": changed_tokens,
                "total_tokens": total_tokens,
                "change_ratio": changed_tokens / total_tokens if total_tokens > 0 else 0,
                "original_tokens": original_tokens,
                "adversarial_tokens": adversarial_tokens
            }
            
            return adversarial_text, projection_info
            
        except Exception as e:
            logger.error(f"Projection to vocab failed: {e}")
            return self.tokenizer.decode(original_ids[0], skip_special_tokens=True), {"error": str(e)}
    
    def attack_batch(self, texts: List[str], method: str = "fgsm", **kwargs) -> List[Tuple[str, float, Dict]]:
        """
        Apply attack to a batch of texts
        
        Args:
            texts: List of input texts
            method: Attack method ('fgsm', 'pgd', 'carlini_wagner')
            **kwargs: Additional arguments for specific attack methods
            
        Returns:
            List of attack results
        """
        results = []
        
        for text in texts:
            try:
                if method == "fgsm":
                    result = self.fgsm_attack(text, **kwargs)
                elif method == "pgd":
                    result = self.pgd_attack(text, **kwargs)
                elif method == "carlini_wagner":
                    result = self.carlini_wagner_attack(text, **kwargs)
                else:
                    raise ValueError(f"Unknown attack method: {method}")
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch attack failed for text: {e}")
                results.append((text, 0.0, {"error": str(e)}))
        
        return results
    
    def evaluate_attack_success(self, original_text: str, adversarial_text: str, 
                              original_prediction: int, adversarial_prediction: int) -> Dict:
        """
        Evaluate attack success metrics
        
        Args:
            original_text: Original input text
            adversarial_text: Adversarial input text
            original_prediction: Original model prediction
            adversarial_prediction: Adversarial model prediction
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Attack success (prediction changed)
        attack_success = original_prediction != adversarial_prediction
        
        # Semantic similarity (using simple word overlap for now)
        original_words = set(original_text.lower().split())
        adversarial_words = set(adversarial_text.lower().split())
        
        if len(original_words) == 0 and len(adversarial_words) == 0:
            semantic_similarity = 1.0
        elif len(original_words) == 0 or len(adversarial_words) == 0:
            semantic_similarity = 0.0
        else:
            intersection = len(original_words.intersection(adversarial_words))
            union = len(original_words.union(adversarial_words))
            semantic_similarity = intersection / union if union > 0 else 0.0
        
        # Edit distance
        from difflib import SequenceMatcher
        edit_similarity = SequenceMatcher(None, original_text, adversarial_text).ratio()
        
        return {
            "attack_success": attack_success,
            "semantic_similarity": semantic_similarity,
            "edit_similarity": edit_similarity,
            "original_prediction": original_prediction,
            "adversarial_prediction": adversarial_prediction
        }
    
    def autoattack(self, text, target_label=None, max_iterations=100):
        """
        AutoAttack - Ensemble of parameter-free attacks
        Combines APGD, FAB, and Square Attack for robust evaluation
        """
        try:
            logger.info("Starting AutoAttack")
            
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Get original prediction
            with torch.no_grad():
                original_logits = self.model(input_ids, attention_mask=attention_mask).logits
                original_pred = torch.argmax(original_logits, dim=-1).item()
            
            if target_label is None:
                target_label = 1 - original_pred
            
            # Run APGD attack
            apgd_result = self._apgd_attack(input_ids, attention_mask, target_label, max_iterations)
            
            # Run FAB attack
            fab_result = self._fab_attack(input_ids, attention_mask, target_label, max_iterations)
            
            # Run Square Attack
            square_result = self._square_attack(input_ids, attention_mask, target_label, max_iterations)
            
            # Select best result
            results = [apgd_result, fab_result, square_result]
            best_result = min(results, key=lambda x: x['perturbation_norm'] if x['success'] else float('inf'))
            
            if best_result['success']:
                adversarial_text = self._generate_adversarial_text(input_ids, best_result['perturbation'])
            else:
                adversarial_text = text
            
            return {
                "success": best_result['success'],
                "original_text": text,
                "adversarial_text": adversarial_text,
                "perturbation_norm": best_result['perturbation_norm'],
                "iterations": max_iterations,
                "attack_type": "autoattack",
                "sub_attacks": {
                    "apgd": apgd_result['success'],
                    "fab": fab_result['success'],
                    "square": square_result['success']
                }
            }
            
        except Exception as e:
            logger.error(f"AutoAttack failed: {e}")
            return {
                "success": False,
                "original_text": text,
                "adversarial_text": text,
                "perturbation_norm": 0,
                "iterations": 0,
                "attack_type": "autoattack",
                "error": str(e)
            }
    
    def deepfool_attack(self, text, target_label=None, max_iterations=50, overshoot=0.02):
        """
        DeepFool Attack
        Minimal perturbation attack that finds the closest decision boundary
        """
        try:
            logger.info("Starting DeepFool attack")
            
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Get original prediction
            with torch.no_grad():
                original_logits = self.model(input_ids, attention_mask=attention_mask).logits
                original_pred = torch.argmax(original_logits, dim=-1).item()
            
            if target_label is None:
                target_label = 1 - original_pred
            
            # Initialize perturbation
            perturbation = torch.zeros_like(input_ids, dtype=torch.float32, requires_grad=True)
            
            for iteration in range(max_iterations):
                perturbation.requires_grad_(True)
                
                # Apply perturbation
                perturbed_embeddings = self.embedding_layer(input_ids) + perturbation.unsqueeze(-1) * 0.1
                
                # Forward pass
                logits = self.model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask).logits
                
                # Check if attack succeeded
                current_pred = torch.argmax(logits, dim=-1).item()
                if current_pred == target_label:
                    break
                
                # Calculate gradients
                target_logit = logits[0, target_label]
                max_other_logit = torch.max(logits[0, :target_label].max(), logits[0, target_label+1:].max())
                
                loss = target_logit - max_other_logit
                loss.backward()
                
                # Calculate minimal perturbation
                grad = perturbation.grad
                grad_norm = torch.norm(grad)
                
                if grad_norm > 0:
                    # DeepFool update rule
                    update = (loss.item() / (grad_norm ** 2)) * grad
                    perturbation = perturbation - update
                else:
                    break
            
            # Apply overshoot
            perturbation = perturbation * (1 + overshoot)
            
            # Generate adversarial text
            adversarial_text = self._generate_adversarial_text(input_ids, perturbation)
            
            # Check success
            with torch.no_grad():
                adv_inputs = self.tokenizer(adversarial_text, return_tensors="pt", padding=True, truncation=True)
                adv_logits = self.model(adv_inputs["input_ids"].to(self.device), 
                                      attention_mask=adv_inputs["attention_mask"].to(self.device)).logits
                adv_pred = torch.argmax(adv_logits, dim=-1).item()
            
            success = adv_pred == target_label
            
            return {
                "success": success,
                "original_text": text,
                "adversarial_text": adversarial_text,
                "perturbation_norm": torch.norm(perturbation).item(),
                "iterations": iteration + 1,
                "attack_type": "deepfool"
            }
            
        except Exception as e:
            logger.error(f"DeepFool attack failed: {e}")
            return {
                "success": False,
                "original_text": text,
                "adversarial_text": text,
                "perturbation_norm": 0,
                "iterations": 0,
                "attack_type": "deepfool",
                "error": str(e)
            }
    
    def mi_fgsm_attack(self, text, target_label=None, epsilon=0.1, max_iterations=10, momentum=0.9):
        """
        Momentum Iterative FGSM (MI-FGSM)
        Improved transferability through momentum
        """
        try:
            logger.info("Starting MI-FGSM attack")
            
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Get original prediction
            with torch.no_grad():
                original_logits = self.model(input_ids, attention_mask=attention_mask).logits
                original_pred = torch.argmax(original_logits, dim=-1).item()
            
            if target_label is None:
                target_label = 1 - original_pred
            
            # Initialize perturbation and momentum
            perturbation = torch.zeros_like(input_ids, dtype=torch.float32)
            momentum_buffer = torch.zeros_like(input_ids, dtype=torch.float32)
            
            for iteration in range(max_iterations):
                perturbation.requires_grad_(True)
                
                # Apply perturbation
                perturbed_embeddings = self.embedding_layer(input_ids) + perturbation.unsqueeze(-1) * 0.1
                
                # Forward pass
                logits = self.model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask).logits
                
                # Calculate loss
                target_logit = logits[0, target_label]
                max_other_logit = torch.max(logits[0, :target_label].max(), logits[0, target_label+1:].max())
                loss = target_logit - max_other_logit
                
                # Calculate gradients
                loss.backward()
                grad = perturbation.grad
                
                # Update momentum
                momentum_buffer = momentum * momentum_buffer + grad / torch.norm(grad, p=1)
                
                # Update perturbation
                perturbation = perturbation + epsilon * torch.sign(momentum_buffer)
                
                # Check success
                current_pred = torch.argmax(logits, dim=-1).item()
                if current_pred == target_label:
                    break
            
            # Generate adversarial text
            adversarial_text = self._generate_adversarial_text(input_ids, perturbation)
            
            # Check final success
            with torch.no_grad():
                adv_inputs = self.tokenizer(adversarial_text, return_tensors="pt", padding=True, truncation=True)
                adv_logits = self.model(adv_inputs["input_ids"].to(self.device), 
                                      attention_mask=adv_inputs["attention_mask"].to(self.device)).logits
                adv_pred = torch.argmax(adv_logits, dim=-1).item()
            
            success = adv_pred == target_label
            
            return {
                "success": success,
                "original_text": text,
                "adversarial_text": adversarial_text,
                "perturbation_norm": torch.norm(perturbation).item(),
                "iterations": iteration + 1,
                "attack_type": "mi_fgsm"
            }
            
        except Exception as e:
            logger.error(f"MI-FGSM attack failed: {e}")
            return {
                "success": False,
                "original_text": text,
                "adversarial_text": text,
                "perturbation_norm": 0,
                "iterations": 0,
                "attack_type": "mi_fgsm",
                "error": str(e)
            }
    
    def di_fgsm_attack(self, text, target_label=None, epsilon=0.1, max_iterations=10, diversity_prob=0.5):
        """
        Diverse Input FGSM (DI-FGSM)
        Better black-box performance through input diversity
        """
        try:
            logger.info("Starting DI-FGSM attack")
            
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Get original prediction
            with torch.no_grad():
                original_logits = self.model(input_ids, attention_mask=attention_mask).logits
                original_pred = torch.argmax(original_logits, dim=-1).item()
            
            if target_label is None:
                target_label = 1 - original_pred
            
            # Initialize perturbation
            perturbation = torch.zeros_like(input_ids, dtype=torch.float32)
            
            for iteration in range(max_iterations):
                perturbation.requires_grad_(True)
                
                # Apply diversity transformation
                if torch.rand(1).item() < diversity_prob:
                    # Random transformation
                    transform = torch.randn_like(input_ids, dtype=torch.float32) * 0.1
                    perturbed_embeddings = self.embedding_layer(input_ids) + (perturbation + transform).unsqueeze(-1) * 0.1
                else:
                    perturbed_embeddings = self.embedding_layer(input_ids) + perturbation.unsqueeze(-1) * 0.1
                
                # Forward pass
                logits = self.model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask).logits
                
                # Calculate loss
                target_logit = logits[0, target_label]
                max_other_logit = torch.max(logits[0, :target_label].max(), logits[0, target_label+1:].max())
                loss = target_logit - max_other_logit
                
                # Calculate gradients
                loss.backward()
                grad = perturbation.grad
                
                # Update perturbation
                perturbation = perturbation + epsilon * torch.sign(grad)
                
                # Check success
                current_pred = torch.argmax(logits, dim=-1).item()
                if current_pred == target_label:
                    break
            
            # Generate adversarial text
            adversarial_text = self._generate_adversarial_text(input_ids, perturbation)
            
            # Check final success
            with torch.no_grad():
                adv_inputs = self.tokenizer(adversarial_text, return_tensors="pt", padding=True, truncation=True)
                adv_logits = self.model(adv_inputs["input_ids"].to(self.device), 
                                      attention_mask=adv_inputs["attention_mask"].to(self.device)).logits
                adv_pred = torch.argmax(adv_logits, dim=-1).item()
            
            success = adv_pred == target_label
            
            return {
                "success": success,
                "original_text": text,
                "adversarial_text": adversarial_text,
                "perturbation_norm": torch.norm(perturbation).item(),
                "iterations": iteration + 1,
                "attack_type": "di_fgsm"
            }
            
        except Exception as e:
            logger.error(f"DI-FGSM attack failed: {e}")
            return {
                "success": False,
                "original_text": text,
                "adversarial_text": text,
                "perturbation_norm": 0,
                "iterations": 0,
                "attack_type": "di_fgsm",
                "error": str(e)
            }
    
    def _apgd_attack(self, input_ids, attention_mask, target_label, max_iterations):
        """APGD (Adaptive PGD) attack implementation"""
        # Simplified APGD implementation
        perturbation = torch.zeros_like(input_ids, dtype=torch.float32, requires_grad=True)
        optimizer = Adam([perturbation], lr=0.01)
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            perturbed_embeddings = self.embedding_layer(input_ids) + perturbation.unsqueeze(-1) * 0.1
            logits = self.model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask).logits
            
            target_logit = logits[0, target_label]
            max_other_logit = torch.max(logits[0, :target_label].max(), logits[0, target_label+1:].max())
            loss = target_logit - max_other_logit
            
            loss.backward()
            optimizer.step()
            
            if torch.argmax(logits, dim=-1).item() == target_label:
                break
        
        return {
            "success": torch.argmax(logits, dim=-1).item() == target_label,
            "perturbation": perturbation,
            "perturbation_norm": torch.norm(perturbation).item()
        }
    
    def _fab_attack(self, input_ids, attention_mask, target_label, max_iterations):
        """FAB (Fast Adaptive Boundary) attack implementation"""
        # Simplified FAB implementation
        perturbation = torch.zeros_like(input_ids, dtype=torch.float32, requires_grad=True)
        
        for iteration in range(max_iterations):
            perturbation.requires_grad_(True)
            
            perturbed_embeddings = self.embedding_layer(input_ids) + perturbation.unsqueeze(-1) * 0.1
            logits = self.model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask).logits
            
            target_logit = logits[0, target_label]
            max_other_logit = torch.max(logits[0, :target_label].max(), logits[0, target_label+1:].max())
            loss = target_logit - max_other_logit
            
            loss.backward()
            
            if perturbation.grad is not None:
                perturbation = perturbation - 0.01 * perturbation.grad
            
            if torch.argmax(logits, dim=-1).item() == target_label:
                break
        
        return {
            "success": torch.argmax(logits, dim=-1).item() == target_label,
            "perturbation": perturbation,
            "perturbation_norm": torch.norm(perturbation).item()
        }
    
    def _square_attack(self, input_ids, attention_mask, target_label, max_iterations):
        """Square Attack implementation"""
        # Simplified Square Attack implementation
        perturbation = torch.zeros_like(input_ids, dtype=torch.float32)
        
        for iteration in range(max_iterations):
            # Random square perturbation
            square_size = max(1, input_ids.shape[1] // 10)
            start_idx = torch.randint(0, input_ids.shape[1] - square_size + 1, (1,)).item()
            
            # Random perturbation
            random_perturbation = torch.randn(square_size) * 0.1
            perturbation[0, start_idx:start_idx + square_size] = random_perturbation
            
            # Check if attack succeeded
            perturbed_embeddings = self.embedding_layer(input_ids) + perturbation.unsqueeze(-1) * 0.1
            logits = self.model(inputs_embeds=perturbed_embeddings, attention_mask=attention_mask).logits
            
            if torch.argmax(logits, dim=-1).item() == target_label:
                break
        
        return {
            "success": torch.argmax(logits, dim=-1).item() == target_label,
            "perturbation": perturbation,
            "perturbation_norm": torch.norm(perturbation).item()
        }