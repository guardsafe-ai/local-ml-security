"""
Genetic Algorithm Implementation
Advanced genetic algorithm for pattern evolution with multi-objective optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
import random
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """Selection methods for genetic algorithm"""
    TOURNAMENT = "tournament"
    RANK = "rank"
    ROULETTE = "roulette"
    ELITIST = "elitist"


class CrossoverMethod(Enum):
    """Crossover methods for genetic algorithm"""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"


class MutationMethod(Enum):
    """Mutation methods for genetic algorithm"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    CUSTOM = "custom"


@dataclass
class Individual:
    """Individual in genetic algorithm population"""
    genes: np.ndarray
    fitness: float
    objectives: List[float]
    age: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for genetic algorithm"""
    population_size: int = 100
    max_generations: int = 1000
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.1
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    tournament_size: int = 5
    convergence_threshold: float = 1e-6
    max_stagnation: int = 50
    diversity_threshold: float = 0.1
    parallel_evaluation: bool = True
    custom_fitness_function: Optional[Callable] = None
    custom_crossover_function: Optional[Callable] = None
    custom_mutation_function: Optional[Callable] = None


class GeneticAlgorithm:
    """
    Advanced Genetic Algorithm Implementation
    Supports multi-objective optimization and various selection/crossover/mutation methods
    """
    
    def __init__(self, config: GeneticAlgorithmConfig):
        """Initialize genetic algorithm"""
        self.config = config
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        self.stagnation_count = 0
        self.converged = False
        
        logger.info("✅ Initialized Genetic Algorithm")
    
    async def evolve(self, 
                    initial_population: Optional[List[Individual]] = None,
                    fitness_function: Optional[Callable] = None) -> Individual:
        """
        Evolve population using genetic algorithm
        """
        try:
            logger.info("Starting genetic algorithm evolution")
            
            # Initialize population
            if initial_population is None:
                await self._initialize_population()
            else:
                self.population = initial_population.copy()
            
            # Set fitness function
            if fitness_function is not None:
                self.config.custom_fitness_function = fitness_function
            
            # Evolution loop
            for generation in range(self.config.max_generations):
                self.generation = generation
                
                # Evaluate fitness
                await self._evaluate_fitness()
                
                # Update best individual
                self._update_best_individual()
                
                # Record fitness history
                self.fitness_history.append(self.best_individual.fitness)
                
                # Check convergence
                if self._check_convergence():
                    logger.info(f"Converged at generation {generation}")
                    break
                
                # Selection
                parents = await self._selection()
                
                # Crossover
                offspring = await self._crossover(parents)
                
                # Mutation
                await self._mutation(offspring)
                
                # Replacement
                await self._replacement(offspring)
                
                # Update diversity
                diversity = self._calculate_diversity()
                self.diversity_history.append(diversity)
                
                # Log progress
                if generation % 10 == 0:
                    logger.info(f"Generation {generation}: Best fitness = {self.best_individual.fitness:.6f}")
            
            logger.info(f"Evolution completed. Best fitness: {self.best_individual.fitness:.6f}")
            return self.best_individual
            
        except Exception as e:
            logger.error(f"Genetic algorithm evolution failed: {e}")
            raise
    
    async def _initialize_population(self):
        """Initialize random population"""
        try:
            self.population = []
            
            for _ in range(self.config.population_size):
                # Generate random genes (this is a simplified example)
                # In practice, you would generate genes based on your specific problem
                genes = np.random.random(10)  # Example: 10-dimensional problem
                
                individual = Individual(
                    genes=genes,
                    fitness=0.0,
                    objectives=[0.0, 0.0]  # Example: 2 objectives
                )
                
                self.population.append(individual)
            
            logger.info(f"Initialized population of {len(self.population)} individuals")
            
        except Exception as e:
            logger.error(f"Population initialization failed: {e}")
            raise
    
    async def _evaluate_fitness(self):
        """Evaluate fitness of all individuals"""
        try:
            if self.config.parallel_evaluation:
                # Parallel evaluation
                tasks = []
                for individual in self.population:
                    task = self._evaluate_individual(individual)
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
            else:
                # Sequential evaluation
                for individual in self.population:
                    await self._evaluate_individual(individual)
            
        except Exception as e:
            logger.error(f"Fitness evaluation failed: {e}")
            raise
    
    async def _evaluate_individual(self, individual: Individual):
        """Evaluate fitness of single individual"""
        try:
            if self.config.custom_fitness_function:
                # Use custom fitness function
                fitness, objectives = await self.config.custom_fitness_function(individual.genes)
                individual.fitness = fitness
                individual.objectives = objectives
            else:
                # Use default fitness function (example)
                individual.fitness = self._default_fitness_function(individual.genes)
                individual.objectives = [individual.fitness, 0.0]  # Single objective
            
        except Exception as e:
            logger.error(f"Individual evaluation failed: {e}")
            individual.fitness = 0.0
            individual.objectives = [0.0, 0.0]
    
    def _default_fitness_function(self, genes: np.ndarray) -> float:
        """Default fitness function (example)"""
        # This is a simple example - in practice, you would implement your specific fitness function
        return np.sum(genes ** 2)  # Minimize sum of squares
    
    async def _selection(self) -> List[Individual]:
        """Select parents for reproduction"""
        try:
            if self.config.selection_method == SelectionMethod.TOURNAMENT:
                return await self._tournament_selection()
            elif self.config.selection_method == SelectionMethod.RANK:
                return await self._rank_selection()
            elif self.config.selection_method == SelectionMethod.ROULETTE:
                return await self._roulette_selection()
            elif self.config.selection_method == SelectionMethod.ELITIST:
                return await self._elitist_selection()
            else:
                raise ValueError(f"Unknown selection method: {self.config.selection_method}")
                
        except Exception as e:
            logger.error(f"Selection failed: {e}")
            raise
    
    async def _tournament_selection(self) -> List[Individual]:
        """Tournament selection"""
        parents = []
        
        for _ in range(self.config.population_size):
            # Select tournament participants
            tournament = random.sample(self.population, self.config.tournament_size)
            
            # Select best individual from tournament
            best = max(tournament, key=lambda x: x.fitness)
            parents.append(best)
        
        return parents
    
    async def _rank_selection(self) -> List[Individual]:
        """Rank-based selection"""
        # Sort population by fitness
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        # Assign ranks
        ranks = np.arange(1, len(sorted_population) + 1)
        
        # Calculate selection probabilities
        probabilities = ranks / np.sum(ranks)
        
        # Select parents based on probabilities
        parents = []
        for _ in range(self.config.population_size):
            selected_index = np.random.choice(len(sorted_population), p=probabilities)
            parents.append(sorted_population[selected_index])
        
        return parents
    
    async def _roulette_selection(self) -> List[Individual]:
        """Roulette wheel selection"""
        # Calculate fitness values
        fitness_values = [individual.fitness for individual in self.population]
        
        # Handle negative fitness values
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1 for f in fitness_values]
        
        # Calculate selection probabilities
        total_fitness = sum(fitness_values)
        probabilities = [f / total_fitness for f in fitness_values]
        
        # Select parents based on probabilities
        parents = []
        for _ in range(self.config.population_size):
            selected_index = np.random.choice(len(self.population), p=probabilities)
            parents.append(self.population[selected_index])
        
        return parents
    
    async def _elitist_selection(self) -> List[Individual]:
        """Elitist selection"""
        # Sort population by fitness
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        # Select elite individuals
        elite_count = int(self.config.population_size * self.config.elitism_rate)
        elite = sorted_population[:elite_count]
        
        # Fill remaining with random selection
        parents = elite.copy()
        while len(parents) < self.config.population_size:
            parents.append(random.choice(self.population))
        
        return parents
    
    async def _crossover(self, parents: List[Individual]) -> List[Individual]:
        """Perform crossover to create offspring"""
        try:
            offspring = []
            
            # Pair parents
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    parent1 = parents[i]
                    parent2 = parents[i + 1]
                    
                    # Perform crossover with probability
                    if random.random() < self.config.crossover_rate:
                        if self.config.crossover_method == CrossoverMethod.SINGLE_POINT:
                            child1, child2 = self._single_point_crossover(parent1, parent2)
                        elif self.config.crossover_method == CrossoverMethod.TWO_POINT:
                            child1, child2 = self._two_point_crossover(parent1, parent2)
                        elif self.config.crossover_method == CrossoverMethod.UNIFORM:
                            child1, child2 = self._uniform_crossover(parent1, parent2)
                        elif self.config.crossover_method == CrossoverMethod.ARITHMETIC:
                            child1, child2 = self._arithmetic_crossover(parent1, parent2)
                        else:
                            raise ValueError(f"Unknown crossover method: {self.config.crossover_method}")
                        
                        offspring.extend([child1, child2])
                    else:
                        # No crossover, copy parents
                        offspring.extend([parent1, parent2])
            
            return offspring
            
        except Exception as e:
            logger.error(f"Crossover failed: {e}")
            raise
    
    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single point crossover"""
        crossover_point = random.randint(1, len(parent1.genes) - 1)
        
        child1_genes = np.concatenate([
            parent1.genes[:crossover_point],
            parent2.genes[crossover_point:]
        ])
        
        child2_genes = np.concatenate([
            parent2.genes[:crossover_point],
            parent1.genes[crossover_point:]
        ])
        
        child1 = Individual(
            genes=child1_genes,
            fitness=0.0,
            objectives=[0.0, 0.0]
        )
        
        child2 = Individual(
            genes=child2_genes,
            fitness=0.0,
            objectives=[0.0, 0.0]
        )
        
        return child1, child2
    
    def _two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two point crossover"""
        point1 = random.randint(1, len(parent1.genes) - 2)
        point2 = random.randint(point1 + 1, len(parent1.genes) - 1)
        
        child1_genes = np.concatenate([
            parent1.genes[:point1],
            parent2.genes[point1:point2],
            parent1.genes[point2:]
        ])
        
        child2_genes = np.concatenate([
            parent2.genes[:point1],
            parent1.genes[point1:point2],
            parent2.genes[point2:]
        ])
        
        child1 = Individual(
            genes=child1_genes,
            fitness=0.0,
            objectives=[0.0, 0.0]
        )
        
        child2 = Individual(
            genes=child2_genes,
            fitness=0.0,
            objectives=[0.0, 0.0]
        )
        
        return child1, child2
    
    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover"""
        child1_genes = np.zeros_like(parent1.genes)
        child2_genes = np.zeros_like(parent2.genes)
        
        for i in range(len(parent1.genes)):
            if random.random() < 0.5:
                child1_genes[i] = parent1.genes[i]
                child2_genes[i] = parent2.genes[i]
            else:
                child1_genes[i] = parent2.genes[i]
                child2_genes[i] = parent1.genes[i]
        
        child1 = Individual(
            genes=child1_genes,
            fitness=0.0,
            objectives=[0.0, 0.0]
        )
        
        child2 = Individual(
            genes=child2_genes,
            fitness=0.0,
            objectives=[0.0, 0.0]
        )
        
        return child1, child2
    
    def _arithmetic_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Arithmetic crossover"""
        alpha = random.random()
        
        child1_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
        child2_genes = (1 - alpha) * parent1.genes + alpha * parent2.genes
        
        child1 = Individual(
            genes=child1_genes,
            fitness=0.0,
            objectives=[0.0, 0.0]
        )
        
        child2 = Individual(
            genes=child2_genes,
            fitness=0.0,
            objectives=[0.0, 0.0]
        )
        
        return child1, child2
    
    async def _mutation(self, offspring: List[Individual]):
        """Apply mutation to offspring"""
        try:
            for individual in offspring:
                if random.random() < self.config.mutation_rate:
                    if self.config.mutation_method == MutationMethod.GAUSSIAN:
                        self._gaussian_mutation(individual)
                    elif self.config.mutation_method == MutationMethod.UNIFORM:
                        self._uniform_mutation(individual)
                    elif self.config.mutation_method == MutationMethod.POLYNOMIAL:
                        self._polynomial_mutation(individual)
                    elif self.config.mutation_method == MutationMethod.CUSTOM:
                        if self.config.custom_mutation_function:
                            await self.config.custom_mutation_function(individual)
                        else:
                            self._gaussian_mutation(individual)
                    else:
                        raise ValueError(f"Unknown mutation method: {self.config.mutation_method}")
            
        except Exception as e:
            logger.error(f"Mutation failed: {e}")
            raise
    
    def _gaussian_mutation(self, individual: Individual):
        """Gaussian mutation"""
        mutation_strength = 0.1  # Adjust as needed
        noise = np.random.normal(0, mutation_strength, individual.genes.shape)
        individual.genes += noise
        
        # Ensure genes stay within bounds (example: [0, 1])
        individual.genes = np.clip(individual.genes, 0, 1)
    
    def _uniform_mutation(self, individual: Individual):
        """Uniform mutation"""
        mutation_strength = 0.1  # Adjust as needed
        noise = np.random.uniform(-mutation_strength, mutation_strength, individual.genes.shape)
        individual.genes += noise
        
        # Ensure genes stay within bounds
        individual.genes = np.clip(individual.genes, 0, 1)
    
    def _polynomial_mutation(self, individual: Individual):
        """Polynomial mutation"""
        eta = 20  # Distribution index
        mutation_strength = 0.1
        
        for i in range(len(individual.genes)):
            if random.random() < 0.5:
                delta = (2 * random.random()) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - random.random())) ** (1 / (eta + 1))
            
            individual.genes[i] += delta * mutation_strength
        
        # Ensure genes stay within bounds
        individual.genes = np.clip(individual.genes, 0, 1)
    
    async def _replacement(self, offspring: List[Individual]):
        """Replace population with offspring"""
        try:
            # Combine parents and offspring
            combined = self.population + offspring
            
            # Sort by fitness
            combined.sort(key=lambda x: x.fitness, reverse=True)
            
            # Select best individuals for next generation
            self.population = combined[:self.config.population_size]
            
        except Exception as e:
            logger.error(f"Replacement failed: {e}")
            raise
    
    def _update_best_individual(self):
        """Update best individual"""
        current_best = max(self.population, key=lambda x: x.fitness)
        
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(current_best)
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
    
    def _check_convergence(self) -> bool:
        """Check if algorithm has converged"""
        # Check stagnation
        if self.stagnation_count >= self.config.max_stagnation:
            return True
        
        # Check fitness improvement
        if len(self.fitness_history) >= 2:
            improvement = abs(self.fitness_history[-1] - self.fitness_history[-2])
            if improvement < self.config.convergence_threshold:
                return True
        
        # Check diversity
        if len(self.diversity_history) > 0:
            if self.diversity_history[-1] < self.config.diversity_threshold:
                return True
        
        return False
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.population) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = np.linalg.norm(
                    self.population[i].genes - self.population[j].genes
                )
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get algorithm statistics"""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": self.best_individual.fitness if self.best_individual else 0.0,
            "average_fitness": np.mean([ind.fitness for ind in self.population]),
            "fitness_std": np.std([ind.fitness for ind in self.population]),
            "diversity": self.diversity_history[-1] if self.diversity_history else 0.0,
            "stagnation_count": self.stagnation_count,
            "converged": self.converged
        }
