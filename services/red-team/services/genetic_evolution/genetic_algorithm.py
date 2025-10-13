"""
Genetic Algorithm for Attack Pattern Evolution
Implements genetic algorithms for evolving attack patterns
"""

import random
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from collections import defaultdict
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class GeneticAttackEvolver:
    """
    Genetic algorithm for evolving attack patterns
    Implements advanced genetic operators for attack optimization
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 10,
                 max_generations: int = 100,
                 fitness_function: Optional[Callable] = None):
        """
        Initialize genetic attack evolver
        
        Args:
            population_size: Size of the population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of elite individuals to preserve
            max_generations: Maximum number of generations
            fitness_function: Custom fitness function
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.fitness_function = fitness_function
        
        # Population and evolution state
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.evolution_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        
        # Genetic operators
        self.crossover_operators = [
            self._single_point_crossover,
            self._two_point_crossover,
            self._uniform_crossover,
            self._arithmetic_crossover
        ]
        
        self.mutation_operators = [
            self._random_mutation,
            self._gaussian_mutation,
            self._swap_mutation,
            self._inversion_mutation
        ]
        
        logger.info(f"✅ Genetic Attack Evolver initialized: pop_size={population_size}, mut_rate={mutation_rate}")
    
    def evolve_attack_patterns(self, 
                             initial_patterns: List[Dict[str, Any]], 
                             target_model: Any = None,
                             evolution_goals: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evolve attack patterns using genetic algorithm
        
        Args:
            initial_patterns: Initial population of attack patterns
            target_model: Target model for evaluation
            evolution_goals: Goals for evolution
            
        Returns:
            Evolution results
        """
        try:
            logger.info(f"Starting evolution with {len(initial_patterns)} initial patterns")
            
            # Initialize population
            self._initialize_population(initial_patterns)
            
            # Evolution loop
            for generation in range(self.max_generations):
                self.generation = generation
                
                # Evaluate fitness
                self._evaluate_fitness(target_model, evolution_goals)
                
                # Record generation statistics
                gen_stats = self._record_generation_stats()
                self.evolution_history.append(gen_stats)
                
                # Check termination criteria
                if self._check_termination_criteria():
                    logger.info(f"Evolution terminated early at generation {generation}")
                    break
                
                # Create next generation
                self._create_next_generation()
                
                # Log progress
                if generation % 10 == 0:
                    logger.info(f"Generation {generation}: best_fitness={self.best_fitness:.4f}, avg_fitness={gen_stats['avg_fitness']:.4f}")
            
            # Final evaluation
            self._evaluate_fitness(target_model, evolution_goals)
            
            # Generate results
            results = self._generate_evolution_results()
            
            logger.info(f"Evolution completed: {len(self.evolution_history)} generations, best_fitness={self.best_fitness:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Attack pattern evolution failed: {e}")
            return {"error": str(e)}
    
    def _initialize_population(self, initial_patterns: List[Dict[str, Any]]) -> None:
        """
        Initialize the population with initial patterns
        
        Args:
            initial_patterns: Initial attack patterns
        """
        try:
            self.population = []
            
            # Add initial patterns
            for pattern in initial_patterns:
                individual = self._create_individual(pattern)
                self.population.append(individual)
            
            # Fill remaining population with random variations
            while len(self.population) < self.population_size:
                base_pattern = random.choice(initial_patterns)
                individual = self._create_individual(base_pattern)
                # Add some random variation
                individual = self._random_variation(individual)
                self.population.append(individual)
            
            # Ensure population size
            self.population = self.population[:self.population_size]
            
            logger.info(f"Initialized population with {len(self.population)} individuals")
            
        except Exception as e:
            logger.error(f"Population initialization failed: {e}")
    
    def _create_individual(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an individual from an attack pattern
        
        Args:
            pattern: Attack pattern
            
        Returns:
            Individual representation
        """
        try:
            individual = {
                'id': f"ind_{len(self.population)}_{int(time.time())}",
                'pattern': pattern,
                'genes': self._extract_genes(pattern),
                'fitness': 0.0,
                'generation_created': self.generation,
                'parent_ids': [],
                'mutation_history': []
            }
            
            return individual
            
        except Exception as e:
            logger.error(f"Individual creation failed: {e}")
            return {'id': 'error', 'pattern': pattern, 'genes': [], 'fitness': 0.0}
    
    def _extract_genes(self, pattern: Dict[str, Any]) -> List[Any]:
        """
        Extract genes from an attack pattern
        
        Args:
            pattern: Attack pattern
            
        Returns:
            List of genes
        """
        try:
            genes = []
            
            # Extract text content
            text = pattern.get('text', pattern.get('pattern', ''))
            genes.append(text)
            
            # Extract categorical features
            category = pattern.get('category', 'unknown')
            genes.append(category)
            
            # Extract numerical features
            severity = pattern.get('severity', 0.5)
            genes.append(severity)
            
            # Extract structural features
            length = len(text)
            genes.append(length)
            
            # Extract word features
            words = text.split()
            word_count = len(words)
            genes.append(word_count)
            
            # Extract character features
            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
            genes.append(special_chars)
            
            return genes
            
        except Exception as e:
            logger.error(f"Gene extraction failed: {e}")
            return []
    
    def _evaluate_fitness(self, target_model: Any = None, evolution_goals: Dict[str, Any] = None) -> None:
        """
        Evaluate fitness of all individuals in the population
        
        Args:
            target_model: Target model for evaluation
            evolution_goals: Goals for evolution
        """
        try:
            self.fitness_scores = []
            
            for individual in self.population:
                fitness = self._calculate_fitness(individual, target_model, evolution_goals)
                individual['fitness'] = fitness
                self.fitness_scores.append(fitness)
                
                # Update best individual
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = individual.copy()
            
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
    
    def _calculate_fitness(self, 
                          individual: Dict[str, Any], 
                          target_model: Any = None, 
                          evolution_goals: Dict[str, Any] = None) -> float:
        """
        Calculate fitness for an individual
        
        Args:
            individual: Individual to evaluate
            target_model: Target model for evaluation
            evolution_goals: Goals for evolution
            
        Returns:
            Fitness score
        """
        try:
            if self.fitness_function:
                return self.fitness_function(individual, target_model, evolution_goals)
            
            # Default fitness calculation
            fitness = 0.0
            
            # Pattern complexity (higher is better for attacks)
            pattern = individual['pattern']
            text = pattern.get('text', pattern.get('pattern', ''))
            
            # Length diversity
            length_score = min(len(text) / 100, 1.0)  # Normalize to [0, 1]
            fitness += length_score * 0.2
            
            # Word diversity
            words = text.split()
            unique_words = len(set(words))
            word_diversity = unique_words / len(words) if words else 0
            fitness += word_diversity * 0.2
            
            # Special character usage
            special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
            special_char_ratio = special_chars / len(text) if text else 0
            fitness += special_char_ratio * 0.2
            
            # Severity score
            severity = pattern.get('severity', 0.5)
            fitness += severity * 0.2
            
            # Novelty score (based on uniqueness)
            novelty = self._calculate_novelty(individual)
            fitness += novelty * 0.2
            
            return fitness
            
        except Exception as e:
            logger.error(f"Fitness calculation failed: {e}")
            return 0.0
    
    def _calculate_novelty(self, individual: Dict[str, Any]) -> float:
        """
        Calculate novelty score for an individual
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Novelty score
        """
        try:
            # Compare with other individuals in population
            text = individual['pattern'].get('text', individual['pattern'].get('pattern', ''))
            
            similarities = []
            for other in self.population:
                if other['id'] != individual['id']:
                    other_text = other['pattern'].get('text', other['pattern'].get('pattern', ''))
                    similarity = self._text_similarity(text, other_text)
                    similarities.append(similarity)
            
            # Novelty is inverse of average similarity
            avg_similarity = np.mean(similarities) if similarities else 0.0
            novelty = 1.0 - avg_similarity
            
            return max(0.0, novelty)
            
        except Exception as e:
            logger.error(f"Novelty calculation failed: {e}")
            return 0.0
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score [0, 1]
        """
        try:
            if not text1 or not text2:
                return 0.0
            
            # Simple Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Text similarity calculation failed: {e}")
            return 0.0
    
    def _create_next_generation(self) -> None:
        """
        Create the next generation using genetic operators
        """
        try:
            new_population = []
            
            # Elitism: preserve best individuals
            elite_individuals = self._select_elite()
            new_population.extend(elite_individuals)
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                if random.random() < self.crossover_rate:
                    offspring1, offspring2 = self._crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.mutation_rate:
                    offspring1 = self._mutate(offspring1)
                if random.random() < self.mutation_rate:
                    offspring2 = self._mutate(offspring2)
                
                # Add offspring
                new_population.extend([offspring1, offspring2])
            
            # Ensure population size
            self.population = new_population[:self.population_size]
            
        except Exception as e:
            logger.error(f"Next generation creation failed: {e}")
    
    def _select_elite(self) -> List[Dict[str, Any]]:
        """
        Select elite individuals for preservation
        
        Returns:
            List of elite individuals
        """
        try:
            # Sort by fitness
            sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
            
            # Select top individuals
            elite = sorted_population[:self.elite_size]
            
            # Update generation info
            for individual in elite:
                individual['generation_created'] = self.generation
            
            return elite
            
        except Exception as e:
            logger.error(f"Elite selection failed: {e}")
            return []
    
    def _tournament_selection(self, tournament_size: int = 3) -> Dict[str, Any]:
        """
        Tournament selection for parent selection
        
        Args:
            tournament_size: Size of tournament
            
        Returns:
            Selected individual
        """
        try:
            # Randomly select tournament participants
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            
            # Select best individual from tournament
            winner = max(tournament, key=lambda x: x['fitness'])
            
            return winner.copy()
            
        except Exception as e:
            logger.error(f"Tournament selection failed: {e}")
            return random.choice(self.population).copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between two parents
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of offspring
        """
        try:
            # Select crossover operator
            crossover_operator = random.choice(self.crossover_operators)
            
            # Perform crossover
            offspring1, offspring2 = crossover_operator(parent1, parent2)
            
            # Update offspring information
            offspring1['parent_ids'] = [parent1['id'], parent2['id']]
            offspring2['parent_ids'] = [parent1['id'], parent2['id']]
            offspring1['generation_created'] = self.generation
            offspring2['generation_created'] = self.generation
            
            return offspring1, offspring2
            
        except Exception as e:
            logger.error(f"Crossover failed: {e}")
            return parent1.copy(), parent2.copy()
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate an individual
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        try:
            # Select mutation operator
            mutation_operator = random.choice(self.mutation_operators)
            
            # Perform mutation
            mutated = mutation_operator(individual)
            
            # Record mutation
            mutated['mutation_history'].append({
                'operator': mutation_operator.__name__,
                'generation': self.generation
            })
            
            return mutated
            
        except Exception as e:
            logger.error(f"Mutation failed: {e}")
            return individual
    
    def _single_point_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single point crossover"""
        try:
            genes1 = parent1['genes'].copy()
            genes2 = parent2['genes'].copy()
            
            if len(genes1) > 1 and len(genes2) > 1:
                crossover_point = random.randint(1, min(len(genes1), len(genes2)) - 1)
                
                # Swap genes after crossover point
                genes1[crossover_point:], genes2[crossover_point:] = genes2[crossover_point:], genes1[crossover_point:]
            
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            offspring1['genes'] = genes1
            offspring2['genes'] = genes2
            
            # Reconstruct pattern from genes
            offspring1['pattern'] = self._reconstruct_pattern(genes1, parent1['pattern'])
            offspring2['pattern'] = self._reconstruct_pattern(genes2, parent2['pattern'])
            
            return offspring1, offspring2
            
        except Exception as e:
            logger.error(f"Single point crossover failed: {e}")
            return parent1.copy(), parent2.copy()
    
    def _two_point_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Two point crossover"""
        try:
            genes1 = parent1['genes'].copy()
            genes2 = parent2['genes'].copy()
            
            if len(genes1) > 2 and len(genes2) > 2:
                point1 = random.randint(0, min(len(genes1), len(genes2)) - 2)
                point2 = random.randint(point1 + 1, min(len(genes1), len(genes2)) - 1)
                
                # Swap genes between points
                genes1[point1:point2], genes2[point1:point2] = genes2[point1:point2], genes1[point1:point2]
            
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            offspring1['genes'] = genes1
            offspring2['genes'] = genes2
            
            offspring1['pattern'] = self._reconstruct_pattern(genes1, parent1['pattern'])
            offspring2['pattern'] = self._reconstruct_pattern(genes2, parent2['pattern'])
            
            return offspring1, offspring2
            
        except Exception as e:
            logger.error(f"Two point crossover failed: {e}")
            return parent1.copy(), parent2.copy()
    
    def _uniform_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Uniform crossover"""
        try:
            genes1 = parent1['genes'].copy()
            genes2 = parent2['genes'].copy()
            
            # Randomly swap each gene
            for i in range(min(len(genes1), len(genes2))):
                if random.random() < 0.5:
                    genes1[i], genes2[i] = genes2[i], genes1[i]
            
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            offspring1['genes'] = genes1
            offspring2['genes'] = genes2
            
            offspring1['pattern'] = self._reconstruct_pattern(genes1, parent1['pattern'])
            offspring2['pattern'] = self._reconstruct_pattern(genes2, parent2['pattern'])
            
            return offspring1, offspring2
            
        except Exception as e:
            logger.error(f"Uniform crossover failed: {e}")
            return parent1.copy(), parent2.copy()
    
    def _arithmetic_crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Arithmetic crossover for numerical genes"""
        try:
            genes1 = parent1['genes'].copy()
            genes2 = parent2['genes'].copy()
            
            # Apply arithmetic crossover to numerical genes
            for i in range(min(len(genes1), len(genes2))):
                if isinstance(genes1[i], (int, float)) and isinstance(genes2[i], (int, float)):
                    alpha = random.random()
                    new_gene1 = alpha * genes1[i] + (1 - alpha) * genes2[i]
                    new_gene2 = (1 - alpha) * genes1[i] + alpha * genes2[i]
                    genes1[i] = new_gene1
                    genes2[i] = new_gene2
            
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            offspring1['genes'] = genes1
            offspring2['genes'] = genes2
            
            offspring1['pattern'] = self._reconstruct_pattern(genes1, parent1['pattern'])
            offspring2['pattern'] = self._reconstruct_pattern(genes2, parent2['pattern'])
            
            return offspring1, offspring2
            
        except Exception as e:
            logger.error(f"Arithmetic crossover failed: {e}")
            return parent1.copy(), parent2.copy()
    
    def _random_mutation(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Random mutation"""
        try:
            mutated = individual.copy()
            genes = mutated['genes'].copy()
            
            # Randomly modify a gene
            if genes:
                gene_index = random.randint(0, len(genes) - 1)
                if isinstance(genes[gene_index], str):
                    # String mutation
                    text = genes[gene_index]
                    if text:
                        char_index = random.randint(0, len(text) - 1)
                        new_char = chr(random.randint(32, 126))  # Printable ASCII
                        genes[gene_index] = text[:char_index] + new_char + text[char_index + 1:]
                elif isinstance(genes[gene_index], (int, float)):
                    # Numerical mutation
                    genes[gene_index] += random.gauss(0, 0.1) * genes[gene_index]
            
            mutated['genes'] = genes
            mutated['pattern'] = self._reconstruct_pattern(genes, individual['pattern'])
            
            return mutated
            
        except Exception as e:
            logger.error(f"Random mutation failed: {e}")
            return individual
    
    def _gaussian_mutation(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Gaussian mutation for numerical genes"""
        try:
            mutated = individual.copy()
            genes = mutated['genes'].copy()
            
            # Apply Gaussian noise to numerical genes
            for i, gene in enumerate(genes):
                if isinstance(gene, (int, float)):
                    noise = random.gauss(0, 0.1)
                    genes[i] = gene + noise * abs(gene)
            
            mutated['genes'] = genes
            mutated['pattern'] = self._reconstruct_pattern(genes, individual['pattern'])
            
            return mutated
            
        except Exception as e:
            logger.error(f"Gaussian mutation failed: {e}")
            return individual
    
    def _swap_mutation(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Swap mutation for string genes"""
        try:
            mutated = individual.copy()
            genes = mutated['genes'].copy()
            
            # Find string genes
            string_indices = [i for i, gene in enumerate(genes) if isinstance(gene, str)]
            
            if len(string_indices) >= 2:
                # Swap two string genes
                idx1, idx2 = random.sample(string_indices, 2)
                genes[idx1], genes[idx2] = genes[idx2], genes[idx1]
            
            mutated['genes'] = genes
            mutated['pattern'] = self._reconstruct_pattern(genes, individual['pattern'])
            
            return mutated
            
        except Exception as e:
            logger.error(f"Swap mutation failed: {e}")
            return individual
    
    def _inversion_mutation(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Inversion mutation for string genes"""
        try:
            mutated = individual.copy()
            genes = mutated['genes'].copy()
            
            # Find string genes
            string_indices = [i for i, gene in enumerate(genes) if isinstance(gene, str)]
            
            if string_indices:
                # Select a string gene
                gene_index = random.choice(string_indices)
                text = genes[gene_index]
                
                if len(text) > 1:
                    # Invert a substring
                    start = random.randint(0, len(text) - 2)
                    end = random.randint(start + 1, len(text))
                    inverted = text[start:end][::-1]
                    genes[gene_index] = text[:start] + inverted + text[end:]
            
            mutated['genes'] = genes
            mutated['pattern'] = self._reconstruct_pattern(genes, individual['pattern'])
            
            return mutated
            
        except Exception as e:
            logger.error(f"Inversion mutation failed: {e}")
            return individual
    
    def _reconstruct_pattern(self, genes: List[Any], original_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct attack pattern from genes
        
        Args:
            genes: List of genes
            original_pattern: Original pattern template
            
        Returns:
            Reconstructed pattern
        """
        try:
            pattern = original_pattern.copy()
            
            if len(genes) >= 1:
                pattern['text'] = str(genes[0])
            if len(genes) >= 2:
                pattern['category'] = str(genes[1])
            if len(genes) >= 3:
                pattern['severity'] = float(genes[2])
            
            return pattern
            
        except Exception as e:
            logger.error(f"Pattern reconstruction failed: {e}")
            return original_pattern
    
    def _random_variation(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add random variation to an individual
        
        Args:
            individual: Individual to vary
            
        Returns:
            Varied individual
        """
        try:
            # Apply random mutation
            return self._mutate(individual)
            
        except Exception as e:
            logger.error(f"Random variation failed: {e}")
            return individual
    
    def _record_generation_stats(self) -> Dict[str, Any]:
        """
        Record statistics for current generation
        
        Returns:
            Generation statistics
        """
        try:
            if not self.fitness_scores:
                return {}
            
            stats = {
                'generation': self.generation,
                'best_fitness': max(self.fitness_scores),
                'worst_fitness': min(self.fitness_scores),
                'avg_fitness': np.mean(self.fitness_scores),
                'std_fitness': np.std(self.fitness_scores),
                'population_size': len(self.population),
                'timestamp': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Generation stats recording failed: {e}")
            return {}
    
    def _check_termination_criteria(self) -> bool:
        """
        Check if evolution should terminate
        
        Returns:
            True if should terminate
        """
        try:
            # Check if fitness has converged
            if len(self.evolution_history) >= 10:
                recent_fitness = [gen['best_fitness'] for gen in self.evolution_history[-10:]]
                if max(recent_fitness) - min(recent_fitness) < 0.01:
                    return True
            
            # Check if maximum generations reached
            if self.generation >= self.max_generations - 1:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Termination criteria check failed: {e}")
            return False
    
    def _generate_evolution_results(self) -> Dict[str, Any]:
        """
        Generate final evolution results
        
        Returns:
            Evolution results
        """
        try:
            # Sort population by fitness
            sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
            
            results = {
                'evolution_completed': True,
                'total_generations': len(self.evolution_history),
                'final_population_size': len(self.population),
                'best_fitness': self.best_fitness,
                'best_individual': self.best_individual,
                'top_individuals': sorted_population[:10],  # Top 10
                'evolution_history': self.evolution_history,
                'final_stats': self.evolution_history[-1] if self.evolution_history else {},
                'timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Evolution results generation failed: {e}")
            return {"error": str(e)}
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """
        Get current evolution status
        
        Returns:
            Current status
        """
        try:
            status = {
                'current_generation': self.generation,
                'population_size': len(self.population),
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean(self.fitness_scores) if self.fitness_scores else 0.0,
                'evolution_progress': self.generation / self.max_generations,
                'best_individual_id': self.best_individual['id'] if self.best_individual else None
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Evolution status retrieval failed: {e}")
            return {"error": str(e)}
    
    def save_evolution_state(self, filepath: str) -> bool:
        """
        Save current evolution state to file
        
        Args:
            filepath: Path to save file
            
        Returns:
            True if successful
        """
        try:
            state = {
                'population': self.population,
                'fitness_scores': self.fitness_scores,
                'generation': self.generation,
                'evolution_history': self.evolution_history,
                'best_individual': self.best_individual,
                'best_fitness': self.best_fitness,
                'parameters': {
                    'population_size': self.population_size,
                    'mutation_rate': self.mutation_rate,
                    'crossover_rate': self.crossover_rate,
                    'elite_size': self.elite_size,
                    'max_generations': self.max_generations
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Evolution state saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Evolution state saving failed: {e}")
            return False
    
    def load_evolution_state(self, filepath: str) -> bool:
        """
        Load evolution state from file
        
        Args:
            filepath: Path to load file
            
        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.population = state.get('population', [])
            self.fitness_scores = state.get('fitness_scores', [])
            self.generation = state.get('generation', 0)
            self.evolution_history = state.get('evolution_history', [])
            self.best_individual = state.get('best_individual')
            self.best_fitness = state.get('best_fitness', float('-inf'))
            
            logger.info(f"Evolution state loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Evolution state loading failed: {e}")
            return False
