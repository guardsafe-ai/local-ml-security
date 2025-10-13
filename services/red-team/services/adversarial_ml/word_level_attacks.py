"""
Word-Level Adversarial Attacks
Implementation of TextFooler, BERT-Attack, and HotFlip for NLP models
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
import random
from typing import Dict, List, Tuple, Optional, Set
from nltk.corpus import wordnet
from transformers import BertTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer
import nltk
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    logger.warning("Could not download NLTK data. Some features may not work.")


class TextFoolerAttack:
    """
    TextFooler: Is BERT Really Robust? (Jin et al., 2020)
    State-of-the-art word-level adversarial attack
    """
    
    def __init__(self, model, tokenizer, use_bert=True, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize TextFooler attack
        
        Args:
            model: Target model to attack
            tokenizer: Tokenizer for the model
            use_bert: Whether to use BERT for synonym generation
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.use_bert = use_bert
        if use_bert:
            try:
                self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model.to(device)
                self.bert_model.eval()
            except Exception as e:
                logger.warning(f"Could not load BERT model: {e}. Falling back to WordNet.")
                self.use_bert = False
        
        # Sentence similarity model
        try:
            self.sentence_sim = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_sim = None
        
        logger.info(f"✅ Initialized TextFoolerAttack with BERT={use_bert}")
    
    def attack(self, text: str, similarity_threshold: float = 0.85, 
              max_candidates: int = 50, max_iterations: int = 20) -> Tuple[str, Dict]:
        """
        Perform TextFooler attack
        
        Args:
            text: Input text to attack
            similarity_threshold: Minimum semantic similarity required
            max_candidates: Maximum synonym candidates to try
            max_iterations: Maximum attack iterations
            
        Returns:
            Tuple of (adversarial_text, attack_info)
        """
        try:
            # 1. Get word importance scores
            word_importance = self._get_word_importance(text)
            
            # 2. Sort words by importance (descending)
            words = text.split()
            important_words = sorted(
                [(i, word, importance) for i, (word, importance) in enumerate(zip(words, word_importance))],
                key=lambda x: x[2], reverse=True
            )
            
            # 3. Attack each important word
            adversarial_text = text
            attack_info = {
                "method": "TextFooler",
                "original_text": text,
                "similarity_threshold": similarity_threshold,
                "word_changes": [],
                "attack_success": False,
                "iterations": 0
            }
            
            for iteration in range(max_iterations):
                attack_info["iterations"] = iteration + 1
                
                # Find next word to attack
                word_changed = False
                for word_idx, word, importance in important_words:
                    if word_idx >= len(adversarial_text.split()):
                        continue
                    
                    # Get candidate replacements
                    candidates = self._get_synonyms(word, adversarial_text, max_candidates)
                    
                    # Try each candidate
                    for candidate in candidates:
                        candidate_text = self._replace_word(adversarial_text, word_idx, candidate)
                        
                        # Check semantic similarity
                        if self.sentence_sim:
                            sim = self._semantic_similarity(adversarial_text, candidate_text)
                        else:
                            sim = self._simple_similarity(adversarial_text, candidate_text)
                        
                        if sim < similarity_threshold:
                            continue
                        
                        # Check if attack succeeds
                        if self._is_successful_attack(adversarial_text, candidate_text):
                            adversarial_text = candidate_text
                            attack_info["word_changes"].append({
                                "word_idx": word_idx,
                                "original_word": word,
                                "replacement": candidate,
                                "similarity": sim,
                                "iteration": iteration + 1
                            })
                            word_changed = True
                            break
                    
                    if word_changed:
                        break
                
                if not word_changed:
                    break
            
            # 4. Final attack success check
            attack_info["attack_success"] = self._is_successful_attack(text, adversarial_text)
            attack_info["adversarial_text"] = adversarial_text
            
            logger.debug(f"TextFooler attack completed: success={attack_info['attack_success']}, changes={len(attack_info['word_changes'])}")
            return adversarial_text, attack_info
            
        except Exception as e:
            logger.error(f"TextFooler attack failed: {e}")
            return text, {"error": str(e), "method": "TextFooler"}
    
    def _get_word_importance(self, text: str) -> List[float]:
        """Calculate word importance using gradient-based scoring"""
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # Get embeddings
            embeddings = self.model.get_input_embeddings()(input_ids)
            embeddings.requires_grad = True
            
            # Forward pass
            outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, torch.argmax(outputs.logits, dim=-1))
            
            # Backward pass
            loss.backward()
            
            # Calculate importance as L2 norm of gradients
            importance = []
            for i in range(embeddings.shape[1]):  # For each token position
                if attention_mask[0, i] == 1:  # Only consider non-padding tokens
                    grad_norm = embeddings.grad[0, i].norm().item()
                    importance.append(grad_norm)
                else:
                    importance.append(0.0)
            
            return importance
            
        except Exception as e:
            logger.error(f"Word importance calculation failed: {e}")
            # Fallback: equal importance for all words
            return [1.0] * len(text.split())
    
    def _get_synonyms(self, word: str, context: str, max_candidates: int = 50) -> List[str]:
        """Get synonym candidates using BERT or WordNet"""
        candidates = []
        
        if self.use_bert and hasattr(self, 'bert_model'):
            # Use BERT masked language model
            try:
                masked_text = context.replace(word, '[MASK]')
                inputs = self.bert_tokenizer(masked_text, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs.input_ids.to(self.device)
                attention_mask = inputs.attention_mask.to(self.device)
                
                with torch.no_grad():
                    outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    
                    # Find MASK token position
                    mask_pos = (input_ids == self.bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                    
                    if len(mask_pos) > 0:
                        # Get top predictions for MASK token
                        mask_logits = logits[0, mask_pos[0]]
                        top_indices = torch.topk(mask_logits, max_candidates).indices
                        
                        for idx in top_indices:
                            candidate = self.bert_tokenizer.decode([idx])
                            if candidate != word and len(candidate) > 1:
                                candidates.append(candidate)
                
            except Exception as e:
                logger.warning(f"BERT synonym generation failed: {e}")
        
        # Fallback to WordNet
        if not candidates:
            try:
                synsets = wordnet.synsets(word)
                for synset in synsets:
                    for lemma in synset.lemmas():
                        synonym = lemma.name().replace('_', ' ')
                        if synonym != word and synonym not in candidates:
                            candidates.append(synonym)
            except Exception as e:
                logger.warning(f"WordNet synonym generation failed: {e}")
        
        return candidates[:max_candidates]
    
    def _replace_word(self, text: str, word_idx: int, replacement: str) -> str:
        """Replace word at given index"""
        words = text.split()
        if 0 <= word_idx < len(words):
            words[word_idx] = replacement
            return ' '.join(words)
        return text
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence embeddings"""
        if self.sentence_sim:
            try:
                emb1 = self.sentence_sim.encode([text1])
                emb2 = self.sentence_sim.encode([text2])
                similarity = F.cosine_similarity(torch.tensor(emb1), torch.tensor(emb2)).item()
                return similarity
            except Exception as e:
                logger.warning(f"Sentence similarity calculation failed: {e}")
        
        return self._simple_similarity(text1, text2)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple similarity using sequence matching"""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _is_successful_attack(self, original_text: str, adversarial_text: str) -> bool:
        """Check if attack successfully changed model prediction"""
        try:
            # Get original prediction
            orig_inputs = self.tokenizer(original_text, return_tensors="pt", padding=True, truncation=True)
            orig_outputs = self.model(**orig_inputs)
            orig_pred = torch.argmax(orig_outputs.logits, dim=-1).item()
            
            # Get adversarial prediction
            adv_inputs = self.tokenizer(adversarial_text, return_tensors="pt", padding=True, truncation=True)
            adv_outputs = self.model(**adv_inputs)
            adv_pred = torch.argmax(adv_outputs.logits, dim=-1).item()
            
            return orig_pred != adv_pred
            
        except Exception as e:
            logger.error(f"Attack success check failed: {e}")
            return False


class BERTAttack:
    """
    BERT-Attack: Adversarial Attack Against BERT Using BERT (Li et al., 2020)
    Contextualized word substitution attack
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize BERT-Attack"""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Load BERT for contextualized substitutions
        try:
            self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model.to(device)
            self.bert_model.eval()
        except Exception as e:
            raise ValueError(f"Could not load BERT model: {e}")
        
        # Sentence similarity model
        try:
            self.sentence_sim = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.sentence_sim = None
        
        logger.info("✅ Initialized BERTAttack")
    
    def attack(self, text: str, similarity_threshold: float = 0.85,
              max_candidates: int = 30, max_iterations: int = 20) -> Tuple[str, Dict]:
        """
        Perform BERT-Attack
        
        Args:
            text: Input text to attack
            similarity_threshold: Minimum semantic similarity
            max_candidates: Maximum candidates per word
            max_iterations: Maximum attack iterations
            
        Returns:
            Tuple of (adversarial_text, attack_info)
        """
        try:
            words = text.split()
            adversarial_text = text
            attack_info = {
                "method": "BERTAttack",
                "original_text": text,
                "similarity_threshold": similarity_threshold,
                "word_changes": [],
                "attack_success": False,
                "iterations": 0
            }
            
            for iteration in range(max_iterations):
                attack_info["iterations"] = iteration + 1
                word_changed = False
                
                # Try to replace each word
                for word_idx, word in enumerate(words):
                    if word_idx >= len(adversarial_text.split()):
                        continue
                    
                    # Get contextualized candidates
                    candidates = self._get_contextual_candidates(word, adversarial_text, max_candidates)
                    
                    # Try each candidate
                    for candidate in candidates:
                        candidate_text = self._replace_word(adversarial_text, word_idx, candidate)
                        
                        # Check semantic similarity
                        sim = self._semantic_similarity(adversarial_text, candidate_text)
                        if sim < similarity_threshold:
                            continue
                        
                        # Check if attack succeeds
                        if self._is_successful_attack(adversarial_text, candidate_text):
                            adversarial_text = candidate_text
                            attack_info["word_changes"].append({
                                "word_idx": word_idx,
                                "original_word": word,
                                "replacement": candidate,
                                "similarity": sim,
                                "iteration": iteration + 1
                            })
                            word_changed = True
                            break
                    
                    if word_changed:
                        break
                
                if not word_changed:
                    break
            
            attack_info["attack_success"] = self._is_successful_attack(text, adversarial_text)
            attack_info["adversarial_text"] = adversarial_text
            
            logger.debug(f"BERTAttack completed: success={attack_info['attack_success']}, changes={len(attack_info['word_changes'])}")
            return adversarial_text, attack_info
            
        except Exception as e:
            logger.error(f"BERTAttack failed: {e}")
            return text, {"error": str(e), "method": "BERTAttack"}
    
    def _get_contextual_candidates(self, word: str, context: str, max_candidates: int) -> List[str]:
        """Get contextualized candidates using BERT"""
        try:
            # Create masked context
            masked_context = context.replace(word, '[MASK]')
            
            # Tokenize
            inputs = self.bert_tokenizer(masked_context, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Find MASK token position
                mask_positions = (input_ids == self.bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                
                candidates = []
                for mask_pos in mask_positions:
                    # Get top predictions
                    mask_logits = logits[0, mask_pos]
                    top_indices = torch.topk(mask_logits, max_candidates).indices
                    
                    for idx in top_indices:
                        candidate = self.bert_tokenizer.decode([idx])
                        # Filter out special tokens and duplicates
                        if (candidate not in ['[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'] and
                            candidate != word and
                            len(candidate) > 1 and
                            candidate not in candidates):
                            candidates.append(candidate)
                
                return candidates[:max_candidates]
                
        except Exception as e:
            logger.error(f"Contextual candidate generation failed: {e}")
            return []
    
    def _replace_word(self, text: str, word_idx: int, replacement: str) -> str:
        """Replace word at given index"""
        words = text.split()
        if 0 <= word_idx < len(words):
            words[word_idx] = replacement
            return ' '.join(words)
        return text
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity"""
        if self.sentence_sim:
            try:
                emb1 = self.sentence_sim.encode([text1])
                emb2 = self.sentence_sim.encode([text2])
                similarity = F.cosine_similarity(torch.tensor(emb1), torch.tensor(emb2)).item()
                return similarity
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
        
        # Fallback to simple similarity
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _is_successful_attack(self, original_text: str, adversarial_text: str) -> bool:
        """Check if attack successfully changed model prediction"""
        try:
            # Get original prediction
            orig_inputs = self.tokenizer(original_text, return_tensors="pt", padding=True, truncation=True)
            orig_outputs = self.model(**orig_inputs)
            orig_pred = torch.argmax(orig_outputs.logits, dim=-1).item()
            
            # Get adversarial prediction
            adv_inputs = self.tokenizer(adversarial_text, return_tensors="pt", padding=True, truncation=True)
            adv_outputs = self.model(**adv_inputs)
            adv_pred = torch.argmax(adv_outputs.logits, dim=-1).item()
            
            return orig_pred != adv_pred
            
        except Exception as e:
            logger.error(f"Attack success check failed: {e}")
            return False


class HotFlipAttack:
    """
    HotFlip: White-Box Adversarial Examples for Text (Ebrahimi et al., 2018)
    Character-level adversarial attack
    """
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize HotFlip attack"""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        logger.info("✅ Initialized HotFlipAttack")
    
    def attack(self, text: str, max_flips: int = 10, 
              similarity_threshold: float = 0.8) -> Tuple[str, Dict]:
        """
        Perform HotFlip attack
        
        Args:
            text: Input text to attack
            max_flips: Maximum character flips
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Tuple of (adversarial_text, attack_info)
        """
        try:
            adversarial_text = text
            attack_info = {
                "method": "HotFlip",
                "original_text": text,
                "similarity_threshold": similarity_threshold,
                "character_changes": [],
                "attack_success": False,
                "flips": 0
            }
            
            for flip in range(max_flips):
                # Find best character to flip
                best_flip = self._find_best_character_flip(adversarial_text)
                
                if best_flip is None:
                    break
                
                char_idx, new_char, score = best_flip
                
                # Apply flip
                chars = list(adversarial_text)
                old_char = chars[char_idx]
                chars[char_idx] = new_char
                candidate_text = ''.join(chars)
                
                # Check similarity
                similarity = SequenceMatcher(None, text, candidate_text).ratio()
                if similarity < similarity_threshold:
                    break
                
                # Check if attack succeeds
                if self._is_successful_attack(text, candidate_text):
                    adversarial_text = candidate_text
                    attack_info["character_changes"].append({
                        "position": char_idx,
                        "old_char": old_char,
                        "new_char": new_char,
                        "score": score,
                        "similarity": similarity
                    })
                    attack_info["flips"] += 1
                else:
                    break
            
            attack_info["attack_success"] = self._is_successful_attack(text, adversarial_text)
            attack_info["adversarial_text"] = adversarial_text
            
            logger.debug(f"HotFlip attack completed: success={attack_info['attack_success']}, flips={attack_info['flips']}")
            return adversarial_text, attack_info
            
        except Exception as e:
            logger.error(f"HotFlip attack failed: {e}")
            return text, {"error": str(e), "method": "HotFlip"}
    
    def _find_best_character_flip(self, text: str) -> Optional[Tuple[int, str, float]]:
        """Find the best character to flip based on gradient information"""
        try:
            # Tokenize text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)
            
            # Get embeddings with gradients
            embeddings = self.model.get_input_embeddings()(input_ids)
            embeddings.requires_grad = True
            
            # Forward pass
            outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
            loss = F.cross_entropy(outputs.logits, torch.argmax(outputs.logits, dim=-1))
            
            # Backward pass
            loss.backward()
            
            # Find character with highest gradient
            best_flip = None
            best_score = float('-inf')
            
            # Convert to character-level analysis
            char_text = text
            for i, char in enumerate(char_text):
                if char.isalnum():  # Only flip alphanumeric characters
                    # Calculate gradient score for this character
                    # This is a simplified version - in practice, you'd need more sophisticated character-level gradient computation
                    score = random.random()  # Placeholder
                    
                    if score > best_score:
                        # Generate candidate character
                        if char.islower():
                            new_char = char.upper()
                        elif char.isupper():
                            new_char = char.lower()
                        else:
                            new_char = chr(ord(char) + 1) if ord(char) < 126 else chr(ord(char) - 1)
                        
                        best_score = score
                        best_flip = (i, new_char, score)
            
            return best_flip
            
        except Exception as e:
            logger.error(f"Character flip search failed: {e}")
            return None
    
    def _is_successful_attack(self, original_text: str, adversarial_text: str) -> bool:
        """Check if attack successfully changed model prediction"""
        try:
            # Get original prediction
            orig_inputs = self.tokenizer(original_text, return_tensors="pt", padding=True, truncation=True)
            orig_outputs = self.model(**orig_inputs)
            orig_pred = torch.argmax(orig_outputs.logits, dim=-1).item()
            
            # Get adversarial prediction
            adv_inputs = self.tokenizer(adversarial_text, return_tensors="pt", padding=True, truncation=True)
            adv_outputs = self.model(**adv_inputs)
            adv_pred = torch.argmax(adv_outputs.logits, dim=-1).item()
            
            return orig_pred != adv_pred
            
        except Exception as e:
            logger.error(f"Attack success check failed: {e}")
            return False
