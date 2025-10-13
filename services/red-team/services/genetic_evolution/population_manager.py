"""
Population Manager for Genetic Evolution
Manages population diversity, selection, and maintenance
"""

import random
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class PopulationManager:
    """
    Manages population for genetic algorithm evolution
    Handles diversity maintenance, selection strategies, and population dynamics
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 diversity_threshold: float = 0.1,
                 max_age: int = 10):
        """
        Initialize population manager
        
        Args:
            population_size: Target population size
            diversity_threshold: Minimum diversity threshold
            max_age: Maximum age of individuals
        """
        self.population_size = population_size
        self.diversity_threshold = diversity_threshold
        self.max_age = max_age
        
        # Population tracking
        self.population = []
        self.generation = 0
        self.diversity_history = []
        self.age_tracking = defaultdict(int)
        
        # Selection strategies
        self.selection_strategies = [
            self._tournament_selection,
            self._roulette_wheel_selection,
            self._rank_selection,
            self._stochastic_universal_sampling
        ]
        
        logger.info(f"✅ Population Manager initialized: size={population_size}, diversity_threshold={diversity_threshold}")
    
    def initialize_population(self, initial_individuals: List[Dict[str, Any]]) -> None:
        """
        Initialize population with initial individuals
        
        Args:
            initial_individuals: List of initial individuals
        """
        try:
            self.population = []
            
            # Add initial individuals
            for individual in initial_individuals:
                individual['id'] = f"ind_{len(self.population)}_{int(datetime.now().timestamp())}"
                individual['generation_created'] = self.generation
                individual['age'] = 0
                self.population.append(individual)
            
            # Fill remaining population with variations
            while len(self.population) < self.population_size:
                base_individual = random.choice(initial_individuals)
                varied_individual = self._create_variation(base_individual)
                varied_individual['id'] = f"ind_{len(self.population)}_{int(datetime.now().timestamp())}"
                varied_individual['generation_created'] = self.generation
                varied_individual['age'] = 0
                self.population.append(varied_individual)
            
            # Ensure population size
            self.population = self.population[:self.population_size]
            
            logger.info(f"Initialized population with {len(self.population)} individuals")
            
        except Exception as e:
            logger.error(f"Population initialization failed: {e}")
    
    def update_generation(self) -> None:
        """Update generation counter and individual ages"""
        try:
            self.generation += 1
            
            # Update individual ages
            for individual in self.population:
                individual['age'] = self.generation - individual['generation_created']
            
            # Remove old individuals
            self._remove_old_individuals()
            
            # Calculate diversity
            diversity = self._calculate_population_diversity()
            self.diversity_history.append({
                'generation': self.generation,
                'diversity': diversity,
                'population_size': len(self.population)
            })
            
            logger.debug(f"Updated to generation {self.generation}, diversity: {diversity:.4f}")
            
        except Exception as e:
            logger.error(f"Generation update failed: {e}")
    
    def select_parents(self, 
                      num_parents: int = 2, 
                      selection_method: str = 'tournament',
                      selection_pressure: float = 1.5) -> List[Dict[str, Any]]:
        """
        Select parents for reproduction
        
        Args:
            num_parents: Number of parents to select
            selection_method: Selection method to use
            selection_pressure: Selection pressure parameter
            
        Returns:
            List of selected parents
        """
        try:
            if len(self.population) < num_parents:
                return self.population.copy()
            
            parents = []
            
            if selection_method == 'tournament':
                for _ in range(num_parents):
                    parent = self._tournament_selection(selection_pressure)
                    parents.append(parent)
            elif selection_method == 'roulette':
                for _ in range(num_parents):
                    parent = self._roulette_wheel_selection()
                    parents.append(parent)
            elif selection_method == 'rank':
                for _ in range(num_parents):
                    parent = self._rank_selection()
                    parents.append(parent)
            elif selection_method == 'sus':
                parents = self._stochastic_universal_sampling(num_parents)
            else:
                # Default to tournament selection
                for _ in range(num_parents):
                    parent = self._tournament_selection(selection_pressure)
                    parents.append(parent)
            
            return parents
            
        except Exception as e:
            logger.error(f"Parent selection failed: {e}")
            return random.sample(self.population, min(num_parents, len(self.population)))
    
    def add_offspring(self, offspring: List[Dict[str, Any]]) -> None:
        """
        Add offspring to population
        
        Args:
            offspring: List of offspring individuals
        """
        try:
            for individual in offspring:
                individual['id'] = f"ind_{len(self.population)}_{int(datetime.now().timestamp())}"
                individual['generation_created'] = self.generation
                individual['age'] = 0
                self.population.append(individual)
            
            # Maintain population size
            self._maintain_population_size()
            
            logger.debug(f"Added {len(offspring)} offspring to population")
            
        except Exception as e:
            logger.error(f"Offspring addition failed: {e}")
    
    def maintain_diversity(self) -> None:
        """
        Maintain population diversity through various strategies
        """
        try:
            # Calculate current diversity
            current_diversity = self._calculate_population_diversity()
            
            if current_diversity < self.diversity_threshold:
                logger.info(f"Low diversity detected: {current_diversity:.4f}, applying diversity maintenance")
                
                # Apply diversity maintenance strategies
                self._apply_diversity_maintenance()
            
        except Exception as e:
            logger.error(f"Diversity maintenance failed: {e}")
    
    def _tournament_selection(self, selection_pressure: float = 1.5) -> Dict[str, Any]:
        """
        Tournament selection
        
        Args:
            selection_pressure: Selection pressure parameter
            
        Returns:
            Selected individual
        """
        try:
            tournament_size = max(2, int(len(self.population) / selection_pressure))
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            
            # Select best individual from tournament
            winner = max(tournament, key=lambda x: x.get('fitness', 0))
            
            return winner.copy()
            
        except Exception as e:
            logger.error(f"Tournament selection failed: {e}")
            return random.choice(self.population).copy()
    
    def _roulette_wheel_selection(self) -> Dict[str, Any]:
        """
        Roulette wheel selection based on fitness
        
        Returns:
            Selected individual
        """
        try:
            # Calculate fitness values
            fitness_values = [individual.get('fitness', 0) for individual in self.population]
            
            # Handle negative fitness values
            min_fitness = min(fitness_values)
            if min_fitness < 0:
                fitness_values = [f - min_fitness + 1 for f in fitness_values]
            
            # Calculate selection probabilities
            total_fitness = sum(fitness_values)
            if total_fitness == 0:
                return random.choice(self.population).copy()
            
            probabilities = [f / total_fitness for f in fitness_values]
            
            # Select individual based on probabilities
            selected_index = np.random.choice(len(self.population), p=probabilities)
            
            return self.population[selected_index].copy()
            
        except Exception as e:
            logger.error(f"Roulette wheel selection failed: {e}")
            return random.choice(self.population).copy()
    
    def _rank_selection(self) -> Dict[str, Any]:
        """
        Rank-based selection
        
        Returns:
            Selected individual
        """
        try:
            # Sort population by fitness
            sorted_population = sorted(self.population, key=lambda x: x.get('fitness', 0))
            
            # Assign ranks (higher fitness = higher rank)
            ranks = list(range(1, len(sorted_population) + 1))
            
            # Calculate selection probabilities based on ranks
            total_rank = sum(ranks)
            probabilities = [r / total_rank for r in ranks]
            
            # Select individual based on rank probabilities
            selected_index = np.random.choice(len(sorted_population), p=probabilities)
            
            return sorted_population[selected_index].copy()
            
        except Exception as e:
            logger.error(f"Rank selection failed: {e}")
            return random.choice(self.population).copy()
    
    def _stochastic_universal_sampling(self, num_individuals: int) -> List[Dict[str, Any]]:
        """
        Stochastic Universal Sampling
        
        Args:
            num_individuals: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        try:
            # Calculate fitness values
            fitness_values = [individual.get('fitness', 0) for individual in self.population]
            
            # Handle negative fitness values
            min_fitness = min(fitness_values)
            if min_fitness < 0:
                fitness_values = [f - min_fitness + 1 for f in fitness_values]
            
            total_fitness = sum(fitness_values)
            if total_fitness == 0:
                return random.sample(self.population, min(num_individuals, len(self.population)))
            
            # Calculate cumulative fitness
            cumulative_fitness = np.cumsum(fitness_values)
            
            # SUS selection
            step = total_fitness / num_individuals
            start = random.uniform(0, step)
            
            selected = []
            for i in range(num_individuals):
                pointer = start + i * step
                
                # Find individual corresponding to pointer
                for j, cum_fit in enumerate(cumulative_fitness):
                    if pointer <= cum_fit:
                        selected.append(self.population[j].copy())
                        break
            
            return selected
            
        except Exception as e:
            logger.error(f"Stochastic universal sampling failed: {e}")
            return random.sample(self.population, min(num_individuals, len(self.population)))
    
    def _calculate_population_diversity(self) -> float:
        """
        Calculate population diversity
        
        Returns:
            Diversity score [0, 1]
        """
        try:
            if len(self.population) < 2:
                return 0.0
            
            # Calculate pairwise distances between individuals
            distances = []
            for i in range(len(self.population)):
                for j in range(i + 1, len(self.population)):
                    distance = self._calculate_individual_distance(
                        self.population[i], 
                        self.population[j]
                    )
                    distances.append(distance)
            
            # Diversity is the average distance
            diversity = np.mean(distances) if distances else 0.0
            
            return diversity
            
        except Exception as e:
            logger.error(f"Diversity calculation failed: {e}")
            return 0.0
    
    def _calculate_individual_distance(self, 
                                     individual1: Dict[str, Any], 
                                     individual2: Dict[str, Any]) -> float:
        """
        Calculate distance between two individuals
        
        Args:
            individual1: First individual
            individual2: Second individual
            
        Returns:
            Distance score [0, 1]
        """
        try:
            # Extract features for comparison
            features1 = self._extract_comparison_features(individual1)
            features2 = self._extract_comparison_features(individual2)
            
            # Calculate distance based on features
            distance = 0.0
            for f1, f2 in zip(features1, features2):
                if isinstance(f1, str) and isinstance(f2, str):
                    # String distance (Jaccard similarity)
                    words1 = set(f1.lower().split())
                    words2 = set(f2.lower().split())
                    if words1 or words2:
                        intersection = len(words1.intersection(words2))
                        union = len(words1.union(words2))
                        distance += 1.0 - (intersection / union)
                    else:
                        distance += 1.0
                elif isinstance(f1, (int, float)) and isinstance(f2, (int, float)):
                    # Numerical distance (normalized)
                    max_val = max(abs(f1), abs(f2), 1.0)
                    distance += abs(f1 - f2) / max_val
                else:
                    # Different types
                    distance += 1.0
            
            # Normalize by number of features
            distance = distance / len(features1) if features1 else 1.0
            
            return min(distance, 1.0)
            
        except Exception as e:
            logger.error(f"Individual distance calculation failed: {e}")
            return 1.0
    
    def _extract_comparison_features(self, individual: Dict[str, Any]) -> List[Any]:
        """
        Extract features for comparison
        
        Args:
            individual: Individual to extract features from
            
        Returns:
            List of features
        """
        try:
            features = []
            
            # Extract pattern features
            pattern = individual.get('pattern', {})
            
            # Text content
            text = pattern.get('text', pattern.get('pattern', ''))
            features.append(text)
            
            # Category
            category = pattern.get('category', 'unknown')
            features.append(category)
            
            # Severity
            severity = pattern.get('severity', 0.5)
            features.append(severity)
            
            # Length
            length = len(text)
            features.append(length)
            
            # Word count
            word_count = len(text.split())
            features.append(word_count)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return []
    
    def _remove_old_individuals(self) -> None:
        """Remove individuals that are too old"""
        try:
            # Filter out old individuals
            young_individuals = [
                ind for ind in self.population 
                if ind['age'] <= self.max_age
            ]
            
            removed_count = len(self.population) - len(young_individuals)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} old individuals")
            
            self.population = young_individuals
            
        except Exception as e:
            logger.error(f"Old individual removal failed: {e}")
    
    def _maintain_population_size(self) -> None:
        """Maintain target population size"""
        try:
            if len(self.population) > self.population_size:
                # Remove excess individuals (keep best ones)
                sorted_population = sorted(
                    self.population, 
                    key=lambda x: x.get('fitness', 0), 
                    reverse=True
                )
                self.population = sorted_population[:self.population_size]
                
            elif len(self.population) < self.population_size:
                # Add random variations to reach target size
                while len(self.population) < self.population_size:
                    base_individual = random.choice(self.population)
                    varied_individual = self._create_variation(base_individual)
                    varied_individual['id'] = f"ind_{len(self.population)}_{int(datetime.now().timestamp())}"
                    varied_individual['generation_created'] = self.generation
                    varied_individual['age'] = 0
                    self.population.append(varied_individual)
            
        except Exception as e:
            logger.error(f"Population size maintenance failed: {e}")
    
    def _create_variation(self, base_individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a variation of an individual
        
        Args:
            base_individual: Base individual to vary
            
        Returns:
            Varied individual
        """
        try:
            varied = base_individual.copy()
            
            # Apply random variation to pattern
            pattern = varied.get('pattern', {})
            text = pattern.get('text', pattern.get('pattern', ''))
            
            if text:
                # Random text variation
                if random.random() < 0.3:  # 30% chance of variation
                    # Add random character
                    char = chr(random.randint(32, 126))
                    text = text + char
                elif random.random() < 0.3:  # 30% chance of variation
                    # Remove random character
                    if len(text) > 1:
                        text = text[:-1]
                elif random.random() < 0.3:  # 30% chance of variation
                    # Replace random character
                    if text:
                        pos = random.randint(0, len(text) - 1)
                        char = chr(random.randint(32, 126))
                        text = text[:pos] + char + text[pos + 1:]
                
                pattern['text'] = text
                varied['pattern'] = pattern
            
            return varied
            
        except Exception as e:
            logger.error(f"Variation creation failed: {e}")
            return base_individual.copy()
    
    def _apply_diversity_maintenance(self) -> None:
        """Apply diversity maintenance strategies"""
        try:
            # Strategy 1: Replace similar individuals with random variations
            self._replace_similar_individuals()
            
            # Strategy 2: Add random individuals
            self._add_random_individuals()
            
            # Strategy 3: Mutate low-diversity individuals
            self._mutate_low_diversity_individuals()
            
        except Exception as e:
            logger.error(f"Diversity maintenance application failed: {e}")
    
    def _replace_similar_individuals(self) -> None:
        """Replace similar individuals with random variations"""
        try:
            # Find pairs of similar individuals
            similar_pairs = []
            for i in range(len(self.population)):
                for j in range(i + 1, len(self.population)):
                    distance = self._calculate_individual_distance(
                        self.population[i], 
                        self.population[j]
                    )
                    if distance < 0.1:  # Very similar
                        similar_pairs.append((i, j, distance))
            
            # Replace one from each similar pair
            for i, j, distance in similar_pairs:
                if i < len(self.population) and j < len(self.population):
                    # Replace the one with lower fitness
                    if self.population[i].get('fitness', 0) < self.population[j].get('fitness', 0):
                        self.population[i] = self._create_variation(self.population[j])
                    else:
                        self.population[j] = self._create_variation(self.population[i])
            
        except Exception as e:
            logger.error(f"Similar individual replacement failed: {e}")
    
    def _add_random_individuals(self) -> None:
        """Add random individuals to increase diversity"""
        try:
            # Add 10% random individuals
            num_random = max(1, len(self.population) // 10)
            
            for _ in range(num_random):
                # Create random individual
                random_individual = self._create_random_individual()
                random_individual['id'] = f"ind_{len(self.population)}_{int(datetime.now().timestamp())}"
                random_individual['generation_created'] = self.generation
                random_individual['age'] = 0
                self.population.append(random_individual)
            
        except Exception as e:
            logger.error(f"Random individual addition failed: {e}")
    
    def _create_random_individual(self) -> Dict[str, Any]:
        """Create a random individual"""
        try:
            # Random text
            text_length = random.randint(10, 100)
            text = ''.join(chr(random.randint(32, 126)) for _ in range(text_length))
            
            # Random pattern
            pattern = {
                'text': text,
                'category': random.choice(['prompt_injection', 'jailbreak', 'system_extraction', 'code_injection']),
                'severity': random.uniform(0.1, 1.0)
            }
            
            individual = {
                'pattern': pattern,
                'fitness': 0.0,
                'genes': [text, pattern['category'], pattern['severity'], len(text), len(text.split())]
            }
            
            return individual
            
        except Exception as e:
            logger.error(f"Random individual creation failed: {e}")
            return {'pattern': {'text': 'random', 'category': 'unknown', 'severity': 0.5}, 'fitness': 0.0}
    
    def _mutate_low_diversity_individuals(self) -> None:
        """Mutate individuals that contribute to low diversity"""
        try:
            # Calculate individual diversity contributions
            diversity_contributions = []
            for i, individual in enumerate(self.population):
                # Calculate average distance to other individuals
                distances = []
                for j, other in enumerate(self.population):
                    if i != j:
                        distance = self._calculate_individual_distance(individual, other)
                        distances.append(distance)
                
                avg_distance = np.mean(distances) if distances else 0.0
                diversity_contributions.append((i, avg_distance))
            
            # Sort by diversity contribution (lowest first)
            diversity_contributions.sort(key=lambda x: x[1])
            
            # Mutate bottom 20% of individuals
            num_to_mutate = max(1, len(self.population) // 5)
            for i in range(num_to_mutate):
                idx = diversity_contributions[i][0]
                self.population[idx] = self._create_variation(self.population[idx])
            
        except Exception as e:
            logger.error(f"Low diversity individual mutation failed: {e}")
    
    def get_population_stats(self) -> Dict[str, Any]:
        """
        Get population statistics
        
        Returns:
            Population statistics
        """
        try:
            if not self.population:
                return {}
            
            fitness_values = [ind.get('fitness', 0) for ind in self.population]
            ages = [ind.get('age', 0) for ind in self.population]
            
            stats = {
                'population_size': len(self.population),
                'generation': self.generation,
                'fitness_stats': {
                    'best': max(fitness_values),
                    'worst': min(fitness_values),
                    'average': np.mean(fitness_values),
                    'std': np.std(fitness_values)
                },
                'age_stats': {
                    'oldest': max(ages),
                    'youngest': min(ages),
                    'average': np.mean(ages)
                },
                'diversity': self._calculate_population_diversity(),
                'diversity_history_length': len(self.diversity_history)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Population stats calculation failed: {e}")
            return {"error": str(e)}
    
    def get_best_individuals(self, num_individuals: int = 10) -> List[Dict[str, Any]]:
        """
        Get best individuals from population
        
        Args:
            num_individuals: Number of best individuals to return
            
        Returns:
            List of best individuals
        """
        try:
            sorted_population = sorted(
                self.population, 
                key=lambda x: x.get('fitness', 0), 
                reverse=True
            )
            
            return sorted_population[:num_individuals]
            
        except Exception as e:
            logger.error(f"Best individuals retrieval failed: {e}")
            return []
    
    def save_population_state(self, filepath: str) -> bool:
        """
        Save current population state
        
        Args:
            filepath: Path to save file
            
        Returns:
            True if successful
        """
        try:
            state = {
                'population': self.population,
                'generation': self.generation,
                'diversity_history': self.diversity_history,
                'parameters': {
                    'population_size': self.population_size,
                    'diversity_threshold': self.diversity_threshold,
                    'max_age': self.max_age
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Population state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Population state saving failed: {e}")
            return False
    
    def load_population_state(self, filepath: str) -> bool:
        """
        Load population state from file
        
        Args:
            filepath: Path to load file
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.population = state.get('population', [])
            self.generation = state.get('generation', 0)
            self.diversity_history = state.get('diversity_history', [])
            
            logger.info(f"Population state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Population state loading failed: {e}")
            return False
